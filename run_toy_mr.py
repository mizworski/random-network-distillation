#!/usr/bin/env python3
import argparse
import functools
import os
import pickle

# from mpi4py import MPI
# import mpi_util
import tf_util
from cmd_util import make_toy_mr_env
from policies.cnn_policy_param_matched import ToyMRCnnPolicy
from ppo_agent import PpoAgent
from utils import set_global_seeds
from vec_env import VecFrameStack


def train(*, env_id, num_env, hps, num_timesteps, seed):
    venv = VecFrameStack(
        make_toy_mr_env(env_id, num_env, seed, env_size=hps.pop('env_size'), wrapper_kwargs=dict(),
                        start_index=num_env,  # * MPI.COMM_WORLD.Get_rank(),
                        max_episode_steps=hps.pop('max_episode_steps')),
        hps.pop('frame_stack'))
    venv.score_multiple = 1
    venv.record_obs = True if env_id == 'SolarisNoFrameskip-v4' else False
    ob_space = venv.observation_space
    ac_space = venv.action_space
    gamma = hps.pop('gamma')
    hps.pop("policy")
    policy = ToyMRCnnPolicy
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
    )
    agent.start_interaction([venv])
    if hps.pop('update_ob_stats_from_random_agent'):
        agent.collect_random_statistics(num_timesteps=128 * 50)
    assert len(hps) == 0, "Unused hyperparameters: %s" % list(hps.keys())

    counter = 0
    while True:
        info = agent.step()
        if info['update']:
            counter += 1
        if agent.I.stats['tcount'] > num_timesteps:
            break

    agent.stop_interaction()


def add_env_params(parser):
    parser.add_argument('--env', help='environment ID', default='toy_mr-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--max_episode_steps', type=int, default=4500)


def main():
    import neptune

    parser = argparse.ArgumentParser(argument_default=None)
    parser.add_argument(
        '--config', action='append',
        help='Gin config files.'
    )
    parser.add_argument(
        '--debug', action='store_true',
        default=False
    )
    cmd_args, unknown = parser.parse_known_args()
    debug = cmd_args.debug
    spec_path = cmd_args.config[0]

    if not debug:
        try:
            with open(spec_path, 'rb') as f:
                import cloudpickle
                specification = cloudpickle.load(f)
        except pickle.UnpicklingError:
            with open(spec_path) as f:
                vars_ = {'script': os.path.basename(spec_path)}
                exec(f.read(), vars_)  # pylint: disable=exec-used
                specification = vars_['experiments_list'][0].to_dict()
                print('NOTE: Only the first experiment from the list will be run!')
        parameters = specification['parameters']
    else:
        print("debug run")
        parameters = dict(env_id="toy_mr", env_size=None)

    class MockArgs(object):
        def add(self, key, value):
            setattr(self, key, value)

    args = MockArgs()

    args.add('env', parameters["env_id"])  # 'chain_env' 'toy_mr'
    args.add('env_size', parameters["env_size"])
    args.add('seed', 0)
    args.add('max_episode_steps', 300)

    args.add('num_timesteps', int(1e12))
    args.add('num_env', 32)
    args.add('use_news', 0)
    args.add('gamma', 0.99)
    args.add('gamma_ext', 0.999)
    args.add('lam', 0.95)
    args.add('update_ob_stats_every_step', 0)
    args.add('update_ob_stats_independently_per_gpu', 0)
    args.add('update_ob_stats_from_random_agent', 1)
    args.add('proportion_of_exp_used_for_predictor_update', 1.)
    args.add('tag', '')
    args.add('policy', 'cnn', )
    args.add('int_coeff', 1.)
    args.add('ext_coeff', 2.)
    args.add('dynamics_bonus', 0)

    if not debug:
        # TODO read more from specification
        print("running with neptune")
        neptune.init(project_qualified_name="pmtest/planning-with-learned-models")
        neptune.create_experiment(name=specification['name'],
                                  tags=specification['tags'],
                                  params=specification['parameters'],
                                  upload_stdout=False,
                                  upload_stderr=False,
                                  )
        neptune.send_metric("test", 777)
        baselines_format_strs = ['log', 'csv']
    else:
        print("running without neptune")
        baselines_format_strs = ['stdout', 'log', 'csv']

    # logger.configure(dir="out", format_strs=baselines_format_strs)

    seed = 10000 * args.seed  # + MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed)

    hps = dict(
        frame_stack=4,
        nminibatches=4,
        nepochs=4,
        lr=0.0001,
        max_grad_norm=0.0,
        env_size=args.env_size,
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
          num_timesteps=args.num_timesteps, hps=hps)


if __name__ == '__main__':
    main()
