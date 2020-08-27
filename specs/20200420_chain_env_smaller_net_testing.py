from mrunner.helpers.specification_helper import create_experiments_helper

experiments_list = create_experiments_helper(
    experiment_name='RND toyMR',
    base_config={
        "idx":0,
        "env_id": "chain_env",
        "env_size": None,
    },
    params_grid={
        "idx": [0],
        "env_size": [15, 20, 25],
    },
    script='python3 -m run_toy_mr --mrunner --output_dir=./out '
           '--config_file=configs/empty.gin',
    exclude=['.pytest_cache', '.vagrant', '__pycache__',
             'checkpoints', 'out', 'Vagrantfile', 'singularity.def',
             'rnd_toyMR_20200417.simg'],
    python_path='alpacka',
    tags=[globals()['script'][:-3]],
    with_neptune=True
)
