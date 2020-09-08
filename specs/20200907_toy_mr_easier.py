from mrunner.helpers.specification_helper import create_experiments_helper

experiments_list = create_experiments_helper(
    experiment_name='rnd_toymr_easier',
    base_config={
        "env_id": "toy_mr",
        "env_size": None,
        'max_episode_steps': 300,

    },
    params_grid={
        "idx": [1],
        'map_file': [
            'four_rooms.txt',
            'hall_way_shifted.txt',
        ],
        "int_coeff": [0, 2, 100],
        # "int_coeff": [2, 1000],
        "lr": [1e-3],
        # "lr": [1e-3],
        "frame_stack": [4],
        'hidsize': [50],
        'nepochs': [1],
        # "frame_stack": [1, 4],
    },
    script='python3 -m run_toy_mr --mrunner --output_dir=./out --config_file=configs/empty.gin',
    exclude=['.pytest_cache', '.vagrant', '__pycache__',
             'checkpoints', 'out', 'Vagrantfile', 'singularity.def',
             'rnd_toyMR_20200417.simg'],
    python_path='',
    tags=[globals()['script'][:-3], 'rnd', 'easier'],
    with_neptune=True
)
