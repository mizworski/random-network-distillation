from mrunner.helpers.specification_helper import create_experiments_helper

experiments_list = create_experiments_helper(
    experiment_name='rnd_atari',
    base_config={
        "idx":0,
    },
    params_grid={
        "idx": [0]
    },
    script='python3 -m run_atari',
    exclude=['.pytest_cache', '.vagrant', '__pycache__',
             'checkpoints', 'out', 'Vagrantfile', 'singularity.def',
             'rnd_toyMR_20200417.simg'],
    python_path='',
    tags=[globals()['script'][:-3]],
    with_neptune=True
)
