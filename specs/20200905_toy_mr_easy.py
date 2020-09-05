from mrunner.helpers.specification_helper import create_experiments_helper

experiments_list = create_experiments_helper(
    experiment_name='rnd_toymr',
    base_config={
        "env_id": "toy_mr",
        "env_size": None,
    },
    params_grid={
        "idx": [0],
        'ToyMR.map_file': ['%ToyMRMaps.ONE_ROOM', '%ToyMRMaps.HALL_WAY', '%ToyMRMaps.FULL_MAP_EASY'],
    },
    script='python3 -m run_toy_mr --mrunner --output_dir=./out --config_file=configs/empty.gin',
    exclude=['.pytest_cache', '.vagrant', '__pycache__',
             'checkpoints', 'out', 'Vagrantfile', 'singularity.def',
             'rnd_toyMR_20200417.simg'],
    python_path='',
    tags=[globals()['script'][:-3]],
    with_neptune=True
)
