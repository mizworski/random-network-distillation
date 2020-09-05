from mrunner.helpers.specification_helper import create_experiments_helper

experiments_list = create_experiments_helper(
    experiment_name='rnd_toymr',
    base_config={
        "env_id": "toy_mr",
        "env_size": None,
        'hidsize': 50,

    },
    params_grid={
        "idx": [0],
        'map_file': [
            'hall_way_shifted.txt',
            'full_mr_map_easy.txt',
        ],
        "int_coeff": [0, 2, 10, 100, 1000],
        "lr": [1e-2, 1e-3, 1e-4],
    },
    script='python3 -m run_toy_mr --mrunner --output_dir=./out --config_file=configs/empty.gin',
    exclude=['.pytest_cache', '.vagrant', '__pycache__',
             'checkpoints', 'out', 'Vagrantfile', 'singularity.def',
             'rnd_toyMR_20200417.simg'],
    python_path='',
    tags=[globals()['script'][:-3], 'rnd'],
    with_neptune=True
)
