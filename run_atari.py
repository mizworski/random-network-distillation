#!/usr/bin/env python3
import functools

import neptune

import mpi_util
import tf_util
from baselines import logger
from cmd_util import make_atari_env, arg_parser
from policies.cnn_gru_policy_dynamics import CnnGruPolicy
from policies.cnn_policy_param_matched import CnnPolicy
from ppo_agent import PpoAgent
from utils import set_global_seeds
from vec_env import VecFrameStack

from collections import defaultdict
# from mpi4py import MPI
import os, numpy as np
import platform
import tensorflow as tf


def sync_from_root(sess, variables, comm=None):
    """
    Send the root node's parameters to every worker.
    Arguments:
      sess: the TensorFlow session.
      variables: all parameter variables including optimizer's
    """
    if comm is None: comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for var in variables:
        if rank == 0:
            comm.bcast(sess.run(var))
        else:
            import tensorflow as tf
            sess.run(tf.assign(var, comm.bcast(None)))


# def gpu_count():
#     """
#     Count the GPUs on this machine.
#     """
#     if shutil.which('nvidia-smi') is None:
#         return 0
#     output = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv'])
#     return max(0, len(output.split(b'\n')) - 2)
#
# def setup_mpi_gpus():
#     """
#     Set CUDA_VISIBLE_DEVICES using MPI.
#     """
#     num_gpus = gpu_count()
#     if num_gpus == 0:
#         return
#     local_rank, _ = get_local_rank_size(MPI.COMM_WORLD)
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank % num_gpus)

def guess_available_gpus(n_gpus=None):
    if n_gpus is not None:
        return list(range(n_gpus))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_divices = os.environ['CUDA_VISIBLE_DEVICES']
        cuda_visible_divices = cuda_visible_divices.split(',')
        return [int(n) for n in cuda_visible_divices]
    if 'RCALL_NUM_GPU' not in os.environ:
        n_gpus = int(os.environ['RCALL_NUM_GPU'])
        return list(range(n_gpus))
    nvidia_dir = '/proc/driver/nvidia/gpus/'
    if os.path.exists(nvidia_dir):
        n_gpus = len(os.listdir(nvidia_dir))
        return list(range(n_gpus))
    raise Exception("Couldn't guess the available gpus on this machine")


def setup_mpi_gpus():
    """
    Set CUDA_VISIBLE_DEVICES using MPI.
    """
    available_gpus = [0]

    node_id = platform.node()
    nodes_ordered_by_rank = MPI.COMM_WORLD.allgather(node_id)
    processes_outranked_on_this_node = [n for n in nodes_ordered_by_rank[:MPI.COMM_WORLD.Get_rank()] if n == node_id]
    local_rank = len(processes_outranked_on_this_node)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[local_rank])


def get_local_rank_size(comm):
    """
    Returns the rank of each process on its machine
    The processes on a given machine will be assigned ranks
        0, 1, 2, ..., N-1,
    where N is the number of processes on this machine.

    Useful if you want to assign one gpu per machine
    """
    this_node = platform.node()
    ranks_nodes = comm.allgather((comm.Get_rank(), this_node))
    node2rankssofar = defaultdict(int)
    local_rank = None
    for (rank, node) in ranks_nodes:
        if rank == comm.Get_rank():
            local_rank = node2rankssofar[node]
        node2rankssofar[node] += 1
    assert local_rank is not None
    return local_rank, node2rankssofar[this_node]


def share_file(comm, path):
    """
    Copies the file from rank 0 to all other ranks
    Puts it in the same place on all machines
    """
    localrank, _ = get_local_rank_size(comm)
    if comm.Get_rank() == 0:
        with open(path, 'rb') as fh:
            data = fh.read()
        comm.bcast(data)
    else:
        data = comm.bcast(None)
        if localrank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as fh:
                fh.write(data)
    comm.Barrier()


def dict_gather_mean(comm, d):
    alldicts = comm.allgather(d)
    size = comm.Get_size()
    k2li = defaultdict(list)
    for d in alldicts:
        for (k, v) in d.items():
            k2li[k].append(v)
    k2mean = {}
    for (k, li) in k2li.items():
        k2mean[k] = np.mean(li, axis=0) if len(li) == size else np.nan
    return k2mean


class MpiAdamOptimizer(tf.train.AdamOptimizer):
    """Adam optimizer that averages gradients across mpi processes."""

    def __init__(self, comm, **kwargs):
        self.comm = comm
        tf.train.AdamOptimizer.__init__(self, **kwargs)

    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = tf.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g, v in grads_and_vars], axis=0)
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(sum(sizes), np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                              for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars


def mpi_mean(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None: comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n + 1, x.dtype)
    localsum[:n] = xsum.ravel()
    localsum[n] = x.shape[axis]
    globalsum = np.zeros_like(localsum)
    comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]


def mpi_moments(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    mean, count = mpi_mean(x, axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(x - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis + 1:]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), comm=None, use_mpi=True):
        self.mean = np.zeros(shape, 'float64')
        self.use_mpi = use_mpi
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm

    def update(self, x):
        if self.use_mpi:
            batch_mean, batch_std, batch_count = mpi_moments(x, axis=0, comm=self.comm)
        else:
            batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

def train(*, env_id, num_env, hps, num_timesteps, seed, use_neptune=False):
    venv = VecFrameStack(
        make_atari_env(env_id, num_env, seed, wrapper_kwargs=dict(),
                       start_index=num_env,
                       max_episode_steps=hps.pop('max_episode_steps')),
        hps.pop('frame_stack'))
    # venv.score_multiple = {'Mario': 500,
    #                        'MontezumaRevengeNoFrameskip-v4': 100,
    #                        'GravitarNoFrameskip-v4': 250,
    #                        'PrivateEyeNoFrameskip-v4': 500,
    #                        'SolarisNoFrameskip-v4': None,
    #                        'VentureNoFrameskip-v4': 200,
    #                        'PitfallNoFrameskip-v4': 100,
    #                        }[env_id]
    venv.score_multiple = 1
    venv.record_obs = True if env_id == 'SolarisNoFrameskip-v4' else False
    ob_space = venv.observation_space
    ac_space = venv.action_space
    gamma = hps.pop('gamma')
    policy = {'rnn': CnnGruPolicy,
              'cnn': CnnPolicy}[hps.pop('policy')]
    agent = PpoAgent(
        scope='ppo',
        ob_space=ob_space,
        ac_space=ac_space,
        stochpol_fn=functools.partial(
            policy,
            scope='pol',
            ob_space=ob_space,
            ac_space=ac_space,
            update_ob_stats_independently_per_gpu=hps.pop('update_ob_stats_independently_per_gpu'),
            proportion_of_exp_used_for_predictor_update=hps.pop('proportion_of_exp_used_for_predictor_update'),
            dynamics_bonus=hps.pop("dynamics_bonus")
        ),
        gamma=gamma,
        gamma_ext=hps.pop('gamma_ext'),
        lam=hps.pop('lam'),
        nepochs=hps.pop('nepochs'),
        nminibatches=hps.pop('nminibatches'),
        lr=hps.pop('lr'),
        cliprange=0.1,
        nsteps=128,
        ent_coef=0.001,
        max_grad_norm=hps.pop('max_grad_norm'),
        use_news=hps.pop("use_news"),
        comm=None,
        update_ob_stats_every_step=hps.pop('update_ob_stats_every_step'),
        int_coeff=hps.pop('int_coeff'),
        ext_coeff=hps.pop('ext_coeff'),
        use_neptune=use_neptune,
    )
    agent.start_interaction([venv])
    if hps.pop('update_ob_stats_from_random_agent'):
        agent.collect_random_statistics(num_timesteps=128 * 50)
    assert len(hps) == 0, "Unused hyperparameters: %s" % list(hps.keys())

    counter = 0
    while True:
        info = agent.step()
        if info['update']:
            logger.logkvs(info['update'], use_neptune)
            logger.dumpkvs()
            counter += 1
        if agent.I.stats['tcount'] > num_timesteps:
            break

    agent.stop_interaction()


def add_env_params(parser):
    parser.add_argument('--env', help='environment ID', default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--max_episode_steps', type=int, default=4500)


def main():
    parser = arg_parser()
    add_env_params(parser)
    parser.add_argument(
        '--config', action='append',
        help='Gin config files.'
    )
    parser.add_argument('--num-timesteps', type=int, default=int(1e12))
    parser.add_argument('--num_env', type=int, default=32)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gamma_ext', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--update_ob_stats_every_step', type=int, default=0)
    parser.add_argument('--update_ob_stats_independently_per_gpu', type=int, default=0)
    parser.add_argument('--update_ob_stats_from_random_agent', type=int, default=1)
    parser.add_argument('--proportion_of_exp_used_for_predictor_update', type=float, default=1.)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--policy', type=str, default='cnn', choices=['cnn', 'rnn'])
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--ext_coeff', type=float, default=2.)
    parser.add_argument('--dynamics_bonus', type=int, default=0)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if not args.debug:
        # TODO read more from specification
        print("running with neptune")
        neptune.init(project_qualified_name="pmtest/planning-with-learned-models")
        neptune.create_experiment(name='rnd_atari',
                                  upload_stdout=False,
                                  upload_stderr=False,
                                  )
        baselines_format_strs = ['log', 'csv']
    else:
        print("running without neptune")
        baselines_format_strs = ['stdout', 'log', 'csv']
    logger.configure(dir=logger.get_dir(), format_strs=baselines_format_strs)

    # setup_mpi_gpus()

    seed = 10000 * args.seed
    set_global_seeds(seed)

    hps = dict(
        frame_stack=4,
        nminibatches=4,
        nepochs=4,
        lr=0.0001,
        max_grad_norm=0.0,
        use_news=args.use_news,
        gamma=args.gamma,
        gamma_ext=args.gamma_ext,
        max_episode_steps=args.max_episode_steps,
        lam=args.lam,
        update_ob_stats_every_step=args.update_ob_stats_every_step,
        update_ob_stats_independently_per_gpu=args.update_ob_stats_independently_per_gpu,
        update_ob_stats_from_random_agent=args.update_ob_stats_from_random_agent,
        proportion_of_exp_used_for_predictor_update=args.proportion_of_exp_used_for_predictor_update,
        policy=args.policy,
        int_coeff=args.int_coeff,
        ext_coeff=args.ext_coeff,
        dynamics_bonus=args.dynamics_bonus
    )

    tf_util.make_session(make_default=True)
    train(env_id=args.env, num_env=args.num_env, seed=seed,
          num_timesteps=args.num_timesteps, hps=hps, use_neptune=(not args.debug))


if __name__ == '__main__':
    main()
