import time
from collections import deque, defaultdict
from copy import copy

import neptune
import numpy as np
import psutil
import tensorflow as tf

import tf_util
# from mpi4py import MPI
from baselines import logger
from console_util import fmt_row
from graph_distance_logging import GraphDistanceLogger
from mpi_util import RunningMeanStd
from recorder import Recorder
from toy_mr import ToyMR
from utils import explained_variance

NO_STATES = ['NO_STATES']


def log_to_neptune(name, value):
    try:
        neptune.log_metric(name, value)
    except neptune.exceptions.Uninitialized:
        print(f"(neptune) {name} {value}")


def log_image_to_neptune(name, image):
    try:
        neptune.log_image(name, image)
    except neptune.exceptions.Uninitialized:
        print(f"(neptune) {name} <image>")


class SemicolonList(list):
    def __str__(self):
        return '[' + ';'.join([str(x) for x in self]) + ']'


class InteractionState(object):
    """
    Parts of the PPOAgent's state that are based on interaction with a single batch of envs
    """

    def __init__(self, ob_space, ac_space, nsteps, gamma, venvs, stochpol, comm):
        self.lump_stride = venvs[0].num_envs
        self.venvs = venvs
        assert all(
            venv.num_envs == self.lump_stride for venv in self.venvs[1:]), 'All venvs should have the same num_envs'
        self.nlump = len(venvs)
        nenvs = self.nenvs = self.nlump * self.lump_stride
        self.reset_counter = 0
        self.env_results = [None] * self.nlump
        self.buf_vpreds_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_vpreds_ext = np.zeros((nenvs, nsteps), np.float32)
        self.buf_nlps = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs_ext = np.zeros((nenvs, nsteps), np.float32)
        self.buf_rews_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_rews_ext = np.zeros((nenvs, nsteps), np.float32)
        self.buf_acs = np.zeros((nenvs, nsteps, *ac_space.shape), ac_space.dtype)
        self.buf_obs = {k: np.zeros(
            [nenvs, nsteps] + stochpol.ph_ob[k].shape.as_list()[2:],
            dtype=stochpol.ph_ob_dtypes[k])
            for k in stochpol.ph_ob_keys}
        self.buf_ob_last = {k: self.buf_obs[k][:, 0, ...].copy() for k in stochpol.ph_ob_keys}
        self.buf_epinfos = [{} for _ in range(self.nenvs)]
        self.buf_news = np.zeros((nenvs, nsteps), np.float32)
        self.buf_ent = np.zeros((nenvs, nsteps), np.float32)
        self.mem_state = stochpol.initial_state(nenvs)
        self.seg_init_mem_state = copy(self.mem_state)  # Memory state at beginning of segment of timesteps
        self.rff_int = RewardForwardFilter(gamma)
        self.rff_rms_int = RunningMeanStd(comm=comm, use_mpi=False)
        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_int_last = self.buf_vpreds_int[:, 0, ...].copy()
        self.buf_vpred_ext_last = self.buf_vpreds_ext[:, 0, ...].copy()
        self.step_count = 0  # counts number of timesteps that you've interacted with this set of environments
        self.t_last_update = time.time()
        self.statlists = defaultdict(lambda: deque([], maxlen=50))  # Count other stats, e.g. optimizer outputs
        self.stats = defaultdict(float)  # Count episodes and timesteps
        self.stats['epcount'] = 0
        self.stats['n_updates'] = 0
        self.stats['tcount'] = 0

    def close(self):
        for venv in self.venvs:
            venv.close()


def dict_gather(comm, d, op='mean'):
    if comm is None: return d
    alldicts = comm.allgather(d)
    size = comm.Get_size()
    k2li = defaultdict(list)
    for d in alldicts:
        for (k, v) in d.items():
            k2li[k].append(v)
    result = {}
    for (k, li) in k2li.items():
        if op == 'mean':
            result[k] = np.mean(li, axis=0)
        elif op == 'sum':
            result[k] = np.sum(li, axis=0)
        elif op == "max":
            result[k] = np.max(li, axis=0)
        else:
            assert 0, op
    return result


class PpoAgent(object):
    envs = None

    def __init__(self, *, scope,
                 ob_space, ac_space,
                 stochpol_fn,
                 nsteps, nepochs=4, nminibatches=1,
                 gamma=0.99,
                 gamma_ext=0.99,
                 lam=0.95,
                 ent_coef=0,
                 cliprange=0.2,
                 max_grad_norm=1.0,
                 vf_coef=1.0,
                 lr=30e-5,
                 adam_hps=None,
                 testing=False,
                 comm=None, comm_train=None, use_news=False,
                 update_ob_stats_every_step=True,
                 int_coeff=None,
                 ext_coeff=None,
                 use_neptune=False,
                 frame_stack=4,
                 env=None,
                 log_heat_maps=True,
                 num_env=32,
                 ):
        self.lr = lr
        self.use_neptune = use_neptune
        self.ext_coeff = ext_coeff
        self.int_coeff = int_coeff
        self.use_news = use_news
        self.update_ob_stats_every_step = update_ob_stats_every_step
        self.abs_scope = (tf.get_variable_scope().name + '/' + scope).lstrip('/')
        self.testing = testing
        # KC modif
        # self.comm_log = MPI.COMM_SELF
        self.comm_log = None
        if comm is not None and comm.Get_size() > 1:
            self.comm_log = comm
            assert not testing or comm.Get_rank() != 0, "Worker number zero can't be testing"
        if comm_train is not None:
            # KC modif
            # self.comm_train, self.comm_train_size = comm_train, comm_train.Get_size()
            self.comm_train = comm_train
        else:
            # self.comm_train, self.comm_train_size = self.comm_log, self.comm_log.Get_size()
            self.comm_train = self.comm_log
        #
        # KC modif
        # self.is_log_leader = self.comm_log.Get_rank() == 0
        # self.is_train_leader = self.comm_train.Get_rank()==0
        self.is_log_leader = True
        self.is_train_leader = True
        with tf.variable_scope(scope):
            self.best_ret = -np.inf
            self.local_best_ret = - np.inf
            self.rooms = []
            self.local_rooms = []
            self.scores = []
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.stochpol = stochpol_fn()
            self.nepochs = nepochs
            self.cliprange = cliprange
            self.nsteps = nsteps
            self.nminibatches = nminibatches
            self.gamma = gamma
            self.gamma_ext = gamma_ext
            self.lam = lam
            self.adam_hps = adam_hps or dict()
            self.ph_adv = tf.placeholder(tf.float32, [None, None])
            self.ph_ret_int = tf.placeholder(tf.float32, [None, None])
            self.ph_ret_ext = tf.placeholder(tf.float32, [None, None])
            self.ph_oldnlp = tf.placeholder(tf.float32, [None, None])
            self.ph_oldvpred = tf.placeholder(tf.float32, [None, None])
            self.ph_lr = tf.placeholder(tf.float32, [])
            self.ph_lr_pred = tf.placeholder(tf.float32, [])
            self.ph_cliprange = tf.placeholder(tf.float32, [])

            # Define loss.
            neglogpac = self.stochpol.pd_opt.neglogp(self.stochpol.ph_ac)
            entropy = tf.reduce_mean(self.stochpol.pd_opt.entropy())
            vf_loss_int = (0.5 * vf_coef) * tf.reduce_mean(tf.square(self.stochpol.vpred_int_opt - self.ph_ret_int))
            vf_loss_ext = (0.5 * vf_coef) * tf.reduce_mean(tf.square(self.stochpol.vpred_ext_opt - self.ph_ret_ext))
            vf_loss = vf_loss_int + vf_loss_ext
            ratio = tf.exp(self.ph_oldnlp - neglogpac)  # p_new / p_old
            negadv = - self.ph_adv
            pg_losses1 = negadv * ratio
            pg_losses2 = negadv * tf.clip_by_value(ratio, 1.0 - self.ph_cliprange, 1.0 + self.ph_cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            ent_loss = (- ent_coef) * entropy
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.ph_oldnlp))
            maxkl = .5 * tf.reduce_max(tf.square(neglogpac - self.ph_oldnlp))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.ph_cliprange)))
            loss = pg_loss + ent_loss + vf_loss + self.stochpol.aux_loss

            # Create optimizer.
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.abs_scope)
            # logger.info("PPO: using MpiAdamOptimizer connected to %i peers" % self.comm_train_size)
            trainer = tf.train.AdamOptimizer(learning_rate=self.ph_lr, **self.adam_hps)
            # trainer = MpiAdamOptimizer(self.comm_train, learning_rate=self.ph_lr, **self.adam_hps)
            grads_and_vars = trainer.compute_gradients(loss, params)
            grads, vars = zip(*grads_and_vars)
            if max_grad_norm:
                _, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            global_grad_norm = tf.global_norm(grads)
            grads_and_vars = list(zip(grads, vars))
            self._train = trainer.apply_gradients(grads_and_vars)

        # Quantities for reporting.
        self._losses = [loss, pg_loss, vf_loss, entropy, clipfrac, approxkl, maxkl, self.stochpol.aux_loss,
                        self.stochpol.feat_var, self.stochpol.max_feat, global_grad_norm]
        self.loss_names = ['tot', 'pg', 'vf', 'ent', 'clipfrac', 'approxkl', 'maxkl', "auxloss", "featvar",
                           "maxfeat", "gradnorm"]
        self.I = None
        self.disable_policy_update = None
        allvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.abs_scope)
        if self.is_log_leader:
            tf_util.display_var_info(allvars)
        tf.get_default_session().run(tf.variables_initializer(allvars))
        # KC modif
        # sync_from_root(tf.get_default_session(), allvars) #Syncs initialization across mpi workers.
        self.t0 = time.time()
        self.global_tcount = 0

        self.episode_observations = []
        self.dones_count = 0
        self.frame_stack = frame_stack
        self.env = env
        if isinstance(self.env, ToyMR):
            self.graph_distance = GraphDistanceLogger(self.env)
        self.single_slice_shape = ob_space.shape[-1] // frame_stack
        self.log_heat_maps = log_heat_maps

    def start_interaction(self, venvs, disable_policy_update=False):
        self.I = InteractionState(ob_space=self.ob_space, ac_space=self.ac_space,
                                  nsteps=self.nsteps, gamma=self.gamma,
                                  venvs=venvs, stochpol=self.stochpol, comm=self.comm_train)
        self.disable_policy_update = disable_policy_update
        self.recorder = Recorder(nenvs=self.I.nenvs, score_multiple=venvs[0].score_multiple)

    def collect_random_statistics(self, num_timesteps):
        # Initializes observation normalization with data from random agent.
        all_ob = []
        for lump in range(self.I.nlump):
            all_ob.append(self.I.venvs[lump].reset())
        for step in range(num_timesteps):
            for lump in range(self.I.nlump):
                acs = np.random.randint(low=0, high=self.ac_space.n, size=(self.I.lump_stride,))
                self.I.venvs[lump].step_async(acs)
                ob, _, _, _ = self.I.venvs[lump].step_wait()
                all_ob.append(ob)
                if len(all_ob) % (128 * self.I.nlump) == 0:
                    ob_ = np.asarray(all_ob).astype(np.float32).reshape((-1, *self.ob_space.shape))
                    self.stochpol.ob_rms.update(ob_[:, :, :, -self.single_slice_shape:])
                    all_ob.clear()

        for lump in range(self.I.nlump):
            self.I.venvs[lump].reset_history()


    def stop_interaction(self):
        self.I.close()
        self.I = None

    @logger.profile("update")
    def update(self):

        # Some logic gathering best ret, rooms etc using MPI.
        # KC modif
        # temp = sum(MPI.COMM_WORLD.allgather(self.local_rooms), [])
        temp = self.local_rooms
        temp = sorted(list(set(temp)))
        self.rooms = temp

        # KC modif
        # temp = sum(MPI.COMM_WORLD.allgather(self.scores), [])
        temp = self.scores
        temp = sorted(list(set(temp)))
        self.scores = temp

        # KC modif
        # temp = sum(MPI.COMM_WORLD.allgather([self.local_best_ret]), [])
        temp = [self.local_best_ret]
        self.best_ret = max(temp)

        # KC modif
        # eprews = MPI.COMM_WORLD.allgather(np.mean(list(self.I.statlists["eprew"])))
        # local_best_rets = MPI.COMM_WORLD.allgather(self.local_best_ret)
        # n_rooms = sum(MPI.COMM_WORLD.allgather([len(self.local_rooms)]), [])
        eprews = [np.mean(list(self.I.statlists["eprew"]))]
        local_best_rets = [self.local_best_ret]
        n_rooms = [len(self.local_rooms)]

        # KC modif
        # if MPI.COMM_WORLD.Get_rank() == 0:
        if True:
            logger.info(f"Rooms visited {self.rooms}")
            # TODO log to neptune
            logger.info(f"Best return {self.best_ret}")
            logger.info(f"Best local return {sorted(local_best_rets)}")
            logger.info(f"eprews {sorted(eprews)}")
            logger.info(f"n_rooms {sorted(n_rooms)}")
            logger.info(f"All scores {sorted(self.scores)}")

        # if self.use_neptune:
            # log_to_neptune(f"other/Best return", self.best_ret)
            # log_to_neptune(f"other/Extrinsic coefficient", self.ext_coeff)
            # log_to_neptune(f"other/Gamma", self.gamma)
            # log_to_neptune(f"other/Gamma ext", self.gamma_ext)
            # log_to_neptune(f"other/Rooms visited", len(self.rooms))

        # Normalize intrinsic rewards.
        rffs_int = np.array([self.I.rff_int.update(rew) for rew in self.I.buf_rews_int.T])
        self.I.rff_rms_int.update(rffs_int.ravel())
        rews_int = self.I.buf_rews_int / np.sqrt(self.I.rff_rms_int.var)
        self.mean_int_rew = np.mean(rews_int)
        self.max_int_rew = np.max(rews_int)

        # Don't normalize extrinsic rewards.
        rews_ext = self.I.buf_rews_ext

        rewmean, rewstd, rewmax = self.I.buf_rews_int.mean(), self.I.buf_rews_int.std(), np.max(self.I.buf_rews_int)

        # Calculate intrinsic returns and advantages.
        lastgaelam = 0
        for t in range(self.nsteps - 1, -1, -1):  # nsteps-2 ... 0
            if self.use_news:
                nextnew = self.I.buf_news[:, t + 1] if t + 1 < self.nsteps else self.I.buf_new_last
            else:
                nextnew = 0.0  # No dones for intrinsic reward.
            nextvals = self.I.buf_vpreds_int[:, t + 1] if t + 1 < self.nsteps else self.I.buf_vpred_int_last
            nextnotnew = 1 - nextnew
            delta = rews_int[:, t] + self.gamma * nextvals * nextnotnew - self.I.buf_vpreds_int[:, t]
            self.I.buf_advs_int[:, t] = lastgaelam = delta + self.gamma * self.lam * nextnotnew * lastgaelam
        rets_int = self.I.buf_advs_int + self.I.buf_vpreds_int # This is target to vpredint

        # Calculate extrinsic returns and advantages.
        lastgaelam = 0
        for t in range(self.nsteps - 1, -1, -1):  # nsteps-2 ... 0
            nextnew = self.I.buf_news[:, t + 1] if t + 1 < self.nsteps else self.I.buf_new_last
            # Use dones for extrinsic reward.
            nextvals = self.I.buf_vpreds_ext[:, t + 1] if t + 1 < self.nsteps else self.I.buf_vpred_ext_last
            nextnotnew = 1 - nextnew
            delta = rews_ext[:, t] + self.gamma_ext * nextvals * nextnotnew - self.I.buf_vpreds_ext[:, t]
            self.I.buf_advs_ext[:, t] = lastgaelam = delta + self.gamma_ext * self.lam * nextnotnew * lastgaelam
        rets_ext = self.I.buf_advs_ext + self.I.buf_vpreds_ext

        # Combine the extrinsic and intrinsic advantages.
        self.I.buf_advs = self.int_coeff * self.I.buf_advs_int + self.ext_coeff * self.I.buf_advs_ext

        # Collects info for reporting.
        info = dict(
            advmean=self.I.buf_advs.mean(),
            advstd=self.I.buf_advs.std(),
            retintmean=rets_int.mean(),  # previously retmean
            retintstd=rets_int.std(),  # previously retstd
            retextmean=rets_ext.mean(),  # previously not there
            retextstd=rets_ext.std(),  # previously not there
            rewintmean_unnorm=rewmean,  # previously rewmean
            rewintmax_unnorm=rewmax,  # previously not there
            rewintmean_norm=self.mean_int_rew,  # previously rewintmean
            rewintmax_norm=self.max_int_rew,  # previously rewintmax
            rewintstd_unnorm=rewstd,  # previously rewstd
            vpredintmean=self.I.buf_vpreds_int.mean(),  # previously vpredmean
            vpredintstd=self.I.buf_vpreds_int.std(),  # previously vrpedstd
            vpredextmean=self.I.buf_vpreds_ext.mean(),  # previously not there
            vpredextstd=self.I.buf_vpreds_ext.std(),  # previously not there
            ev_int=np.clip(explained_variance(self.I.buf_vpreds_int.ravel(), rets_int.ravel()), -1, None),
            ev_ext=np.clip(explained_variance(self.I.buf_vpreds_ext.ravel(), rets_ext.ravel()), -1, None),
            rooms=SemicolonList(self.rooms),
            n_rooms_hisotry=len(self.rooms),
            best_ret=self.best_ret,
            reset_counter=self.I.reset_counter
        )

        info[f'mem_available'] = psutil.virtual_memory().available

        to_record = {'acs': self.I.buf_acs,
                     'rews_int': self.I.buf_rews_int,
                     'rews_int_norm': rews_int,
                     'rews_ext': self.I.buf_rews_ext,
                     'vpred_int': self.I.buf_vpreds_int,
                     'vpred_ext': self.I.buf_vpreds_ext,
                     'adv_int': self.I.buf_advs_int,
                     'adv_ext': self.I.buf_advs_ext,
                     'ent': self.I.buf_ent,
                     'ret_int': rets_int,
                     'ret_ext': rets_ext,
                     }
        if self.I.venvs[0].record_obs:
            to_record['obs'] = self.I.buf_obs[None]
        self.recorder.record(bufs=to_record,
                             # infos=self.I.buf_epinfos)
                             infos=[])

        # Create feeddict for optimization.
        envsperbatch = self.I.nenvs // self.nminibatches
        ph_buf = [
            (self.stochpol.ph_ac, self.I.buf_acs),
            (self.ph_ret_int, rets_int), # Target to vpredint
            (self.ph_ret_ext, rets_ext), # Target to vpredext
            (self.ph_oldnlp, self.I.buf_nlps),
            (self.ph_adv, self.I.buf_advs),
        ]
        if self.I.mem_state is not NO_STATES:
            ph_buf.extend([
                (self.stochpol.ph_istate, self.I.seg_init_mem_state),
                (self.stochpol.ph_new, self.I.buf_news),
            ])

        verbose = True
        if verbose and self.is_log_leader:
            samples = np.prod(self.I.buf_advs.shape)
            # logger.info("buffer shape %s, samples_per_mpi=%i, mini_per_mpi=%i, samples=%i, mini=%i " % (
            #         str(self.I.buf_advs.shape),
            #         samples, samples//self.nminibatches,
            #         samples*self.comm_train_size, samples*self.comm_train_size//self.nminibatches))
            logger.info(" " * 6 + fmt_row(13, self.loss_names))

        epoch = 0
        start = 0
        # Optimizes on current data for several epochs.
        while epoch < self.nepochs:
            end = start + envsperbatch
            mbenvinds = slice(start, end, None)

            fd = {ph: buf[mbenvinds] for (ph, buf) in ph_buf}
            fd.update({self.ph_lr: self.lr, self.ph_cliprange: self.cliprange})
            fd[self.stochpol.ph_ob[None]] = np.concatenate(
                [self.I.buf_obs[None][mbenvinds], self.I.buf_ob_last[None][mbenvinds, None]], 1)
            assert list(fd[self.stochpol.ph_ob[None]].shape) == [self.I.nenvs // self.nminibatches,
                                                                 self.nsteps + 1] + list(self.ob_space.shape), \
                [fd[self.stochpol.ph_ob[None]].shape,
                 [self.I.nenvs // self.nminibatches, self.nsteps + 1] + list(self.ob_space.shape)]
            fd.update({self.stochpol.ph_mean: self.stochpol.ob_rms.mean,
                       self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5})

            ret = tf.get_default_session().run(self._losses + [self._train], feed_dict=fd)[:-1]
            if not self.testing:
                lossdict = dict(zip([n for n in self.loss_names], ret), axis=0)
            else:
                lossdict = {}
            # Synchronize the lossdict across mpi processes, otherwise weights may be rolled back on one process but not another.
            _maxkl = lossdict.pop('maxkl')
            lossdict = dict_gather(self.comm_train, lossdict, op='mean')
            maxmaxkl = dict_gather(self.comm_train, {"maxkl": _maxkl}, op='max')
            lossdict["maxkl"] = maxmaxkl["maxkl"]
            if verbose and self.is_log_leader:
                logger.info("%i:%03i %s" % (epoch, start, fmt_row(13, [lossdict[n] for n in self.loss_names])))
                if self.use_neptune:
                    for metric_name, metric_value in lossdict.items():
                        neptune.log_metric(f"rnd/{metric_name}", metric_value)
            start += envsperbatch
            if start == self.I.nenvs:
                epoch += 1
                start = 0

        if self.is_train_leader:
            self.I.stats["n_updates"] += 1
            info.update([('opt_' + n, lossdict[n]) for n in self.loss_names])
            tnow = time.time()
            info['tps'] = self.nsteps * self.I.nenvs / (tnow - self.I.t_last_update)
            info['time_elapsed'] = time.time() - self.t0
            self.I.t_last_update = tnow
        self.stochpol.update_normalization(
            # Necessary for continuous control tasks with odd obs ranges, only implemented in mlp policy,
            ob=self.I.buf_obs  # NOTE: not shared via MPI
        )
        return info

    def env_step(self, l, acs):
        self.I.venvs[l].step_async(acs)
        self.I.env_results[l] = None

    def env_get(self, l):
        """
        Get most recent (obs, rews, dones, infos) from vectorized environment
        Using step_wait if necessary
        """
        if self.I.step_count == 0:  # On the zeroth step with a new venv, we need to call reset on the environment
            ob = self.I.venvs[l].reset()
            out = self.I.env_results[l] = (ob, None, np.ones(self.I.lump_stride, bool), {})
        else:
            if self.I.env_results[l] is None:
                out = self.I.env_results[l] = self.I.venvs[l].step_wait()
            else:
                out = self.I.env_results[l]
        return out

    def log_heat_map(self, episode_observations):
        state_visit_freq = {}
        invalid_transitions = {}
        for observation in episode_observations:
            state = self.env.obs2state(observation)
            if state not in state_visit_freq:
                state_visit_freq[state] = 0
            state_visit_freq[state] += 1

        visit_heat_map = self.env.render_visit_heat_map(
            state_visit_freq, invalid_transitions, separate_by_keys=False
        )
        log_image_to_neptune(f'heat_maps', visit_heat_map)

    @logger.profile("step")
    def step(self):
        # Does a rollout.
        t = self.I.step_count % self.nsteps
        epinfos = []
        episodes_visited_rooms = []
        episodes_keys_taken = []
        episodes_doors_opened = []
        states_visited_in_epoch = []
        states_visited_in_history = []
        max_distance_from_start_in_history = []
        min_distance_to_goal_in_history = []
        for l in range(self.I.nlump):
            obs, prevrews, news, infos = self.env_get(l)
            obs_len = obs.shape[-1] // self.frame_stack
            obs_batch = obs[:, ..., -obs_len:]
            from toy_mr import ToyMR
            if isinstance(self.env, ToyMR):
                for obs_batch_i in obs_batch:
                    self.graph_distance.update_distances(self.env.obs2state(obs_batch_i))

            if news[0]:
                self.dones_count += 1
            if self.log_heat_maps and self.dones_count % 20 == 0:
                self.log_heat_map(np.stack(self.episode_observations))
                self.episode_observations = []
                self.dones_count += 1
            elif self.dones_count % 20 == 19:
                obs_len = obs.shape[-1] // self.frame_stack
                self.episode_observations.append(obs[0, 0, 0, -obs_len:])

            for env_pos_in_lump, (ob, info, done) in enumerate(zip(obs, infos, news)):
                if done:
                    info.update(self.env.calculate_statistics(ob[..., -obs_len:]))
                    if isinstance(self.env, ToyMR):
                        info.update(self.graph_distance.result())
                if 'episode' in info:
                    # Information like rooms visited is added to info on end of episode.
                    epinfos.append(info['episode'])
                    info_with_places = info['episode']
                    try:
                        # KC modif
                        # info_with_places['places'] = info['episode']['visited_rooms']
                        info_with_places['places'] = []
                    except:
                        import ipdb;
                        ipdb.set_trace()
                    self.I.buf_epinfos[env_pos_in_lump + l * self.I.lump_stride][t] = info_with_places
                if 'room_first_visit' in info and done:
                    visited_rooms = [
                        room_loc for room_loc, first_visit in info['room_first_visit'].items()
                        if first_visit is not None
                    ]
                    # self.I.buf_epinfos[env_pos_in_lump+l*self.I.lump_stride][t] = {
                    #     'visited_rooms': visited_rooms
                    # }
                    episodes_visited_rooms.append(visited_rooms)
                    episodes_keys_taken.append(info['nb_keys_taken'])
                    episodes_doors_opened.append(info['nb_doors_opened'])
                if 'visited_states_in_episode' in info and done:
                    states_visited_in_epoch.append(info['visited_states_in_episode'])
                if 'visited_states_in_history' in info and done:
                    states_visited_in_history.append(info['visited_states_in_history'])
                if 'max_distance_from_start_in_history' in info and done:
                    max_distance_from_start_in_history.append(info['max_distance_from_start_in_history'])
                if 'min_distance_to_goal_in_history' in info and done:
                    min_distance_to_goal_in_history.append(info['min_distance_to_goal_in_history'])

            sli = slice(l * self.I.lump_stride, (l + 1) * self.I.lump_stride)
            memsli = slice(None) if self.I.mem_state is NO_STATES else sli
            dict_obs = self.stochpol.ensure_observation_is_dict(obs)
            # TODO kc modif
            # with logger.ProfileKV("policy_inference"):
            with logger.profile_kv("policy_inference"):
                # Calls the policy and value function on current observation.
                acs, vpreds_int, vpreds_ext, nlps, self.I.mem_state[memsli], ent = self.stochpol.call(
                    dict_obs, news, self.I.mem_state[memsli], update_obs_stats=self.update_ob_stats_every_step
                )
            self.env_step(l, acs)

            # Update buffer with transition.
            for k in self.stochpol.ph_ob_keys:
                self.I.buf_obs[k][sli, t] = dict_obs[k]
            self.I.buf_news[sli, t] = news
            self.I.buf_vpreds_int[sli, t] = vpreds_int
            self.I.buf_vpreds_ext[sli, t] = vpreds_ext
            self.I.buf_nlps[sli, t] = nlps
            self.I.buf_acs[sli, t] = acs
            self.I.buf_ent[sli, t] = ent

            if t > 0:
                self.I.buf_rews_ext[sli, t - 1] = prevrews

        self.I.step_count += 1
        if t == self.nsteps - 1 and not self.disable_policy_update:
            # We need to take one extra step so every transition has a reward.
            for l in range(self.I.nlump):
                sli = slice(l * self.I.lump_stride, (l + 1) * self.I.lump_stride)
                memsli = slice(None) if self.I.mem_state is NO_STATES else sli
                nextobs, rews, nextnews, _ = self.env_get(l)
                dict_nextobs = self.stochpol.ensure_observation_is_dict(nextobs)
                for k in self.stochpol.ph_ob_keys:
                    self.I.buf_ob_last[k][sli] = dict_nextobs[k]
                self.I.buf_new_last[sli] = nextnews
                # TODO kc modif
                # with logger.ProfileKV("policy_inference"):
                with logger.profile_kv("policy_inference"):
                    _, self.I.buf_vpred_int_last[sli], self.I.buf_vpred_ext_last[sli], _, _, _ = self.stochpol.call(
                        dict_nextobs, nextnews, self.I.mem_state[memsli], update_obs_stats=False)
                self.I.buf_rews_ext[sli, t] = rews

            # Calcuate the intrinsic rewards for the rollout.
            fd = {}
            ob_data = np.concatenate([self.I.buf_obs[None], self.I.buf_ob_last[None][:, None]], 1)
            fd[self.stochpol.ph_ob[None]] = ob_data
            fd.update({self.stochpol.ph_mean: self.stochpol.ob_rms.mean,
                       self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5})
            fd[self.stochpol.ph_ac] = self.I.buf_acs
            self.I.buf_rews_int[:] = tf.get_default_session().run(self.stochpol.int_rew, fd)

            if not self.update_ob_stats_every_step:
                # Update observation normalization parameters after the rollout is completed.
                obs_ = self.I.buf_obs[None].astype(np.float32)
                self.stochpol.ob_rms.update(obs_.reshape((-1, *obs_.shape[2:]))[:, :, :, -1:])
            if not self.testing:
                update_info = self.update()
            else:
                update_info = {}
            self.I.seg_init_mem_state = copy(self.I.mem_state)
            global_i_stats = dict_gather(self.comm_log, self.I.stats, op='sum')
            global_deque_mean = dict_gather(self.comm_log, {n: np.mean(dvs) for n, dvs in self.I.statlists.items()},
                                            op='mean')
            global_deque_max = dict_gather(
                self.comm_log, {f"{n}_max": max(dvs) for n, dvs in self.I.statlists.items() if dvs}, op='max')
            global_deque_min = dict_gather(
                self.comm_log, {f"{n}_min": min(dvs) for n, dvs in self.I.statlists.items() if dvs}, op='min')
            update_info.update(global_i_stats)
            update_info.update(global_deque_mean)
            update_info.update(global_deque_max)
            update_info.update(global_deque_min)
            self.global_tcount = global_i_stats['tcount']
            for infos_ in self.I.buf_epinfos:
                infos_.clear()
        else:
            update_info = {}

        # Some reporting logic.
        for visited_rooms in episodes_visited_rooms:
            self.local_rooms += list(visited_rooms)
            self.local_rooms = sorted(list(set(self.local_rooms)))
            self.I.statlists['eprooms'].append(len(visited_rooms))
        if episodes_keys_taken:
            self.I.statlists['epkeys'].extend(episodes_keys_taken)
        if episodes_doors_opened:
            self.I.statlists['epdoors'].extend(episodes_doors_opened)
        if states_visited_in_epoch:
            self.I.statlists['visited_states_in_episode'].extend(states_visited_in_epoch)

        if states_visited_in_history:
            self.I.statlists['visited_states_in_history'].extend(states_visited_in_history)
        if max_distance_from_start_in_history:
            self.I.statlists['max_distance_from_start_in_history'].extend(max_distance_from_start_in_history)
        if min_distance_to_goal_in_history:
            self.I.statlists['min_distance_to_goal_in_history'].extend(min_distance_to_goal_in_history)

        for epinfo in epinfos:
            if self.testing:
                self.I.statlists['eprew_test'].append(epinfo['r'])
                self.I.statlists['eplen_test'].append(epinfo['l'])
            else:
                self.I.statlists['eprew'].append(epinfo['r'])
                if self.local_best_ret is None:
                    self.local_best_ret = epinfo["r"]
                elif epinfo["r"] > self.local_best_ret:
                    self.local_best_ret = epinfo["r"]

                self.I.statlists['eplen'].append(epinfo['l'])
                self.I.stats['epcount'] += 1
                self.I.stats['tcount'] += epinfo['l']
                self.I.stats['rewtotal'] += epinfo['r']
                self.I.stats["best_ext_ret"] = self.best_ret

        return {'update': update_info}


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
