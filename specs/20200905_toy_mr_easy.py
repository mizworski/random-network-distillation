from mrunner.helpers.specification_helper import create_experiments_helper

experiments_list = create_experiments_helper(
    experiment_name='rnd_toymr_easy',
    base_config={
        "env_id": "toy_mr",
        "env_size": None,
        'hidsize': 50,
        'max_episode_steps': 600,

    },
    params_grid={
        "idx": [0],
        'map_file': [
            'full_mr_map.txt',
        ],
        "int_coeff": [0, 2, 10, 1000],
        # "int_coeff": [2, 1000],
        "lr": [1e-2, 1e-4],
        # "lr": [1e-3],
        "frame_stack": [4],
        # "frame_stack": [1, 4],
    },
    script='python3 -m run_toy_mr --mrunner --output_dir=./out --config_file=configs/empty.gin',
    exclude=['.pytest_cache', '.vagrant', '__pycache__',
             'checkpoints', 'out', 'Vagrantfile', 'singularity.def',
             'rnd_toyMR_20200417.simg'],
    python_path='',
    tags=[globals()['script'][:-3], 'rnd', 'full_easy'],
    with_neptune=True
)
