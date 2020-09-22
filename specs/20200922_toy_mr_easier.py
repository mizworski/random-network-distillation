from mrunner.helpers.specification_helper import create_experiments_helper

experiments_list = create_experiments_helper(
    experiment_name='rnd_toymr_easy',
    base_config={
        "env_id": "toy_mr",
        "env_size": None,
        "lr": 1e-4,
        "hidsize": 64,
        "predictor_hid_size": 512,
        "update_ob_stats_every_step": False,
        "frame_stack": 4,
        'nepochs': 16,
        'vf_coeff': 1,
        'int_coeff': 10,
        "ext_coeff": 1,
        'lam': 0.95,
        'gamma': 0.99,
        'gamma_ext': 0.999,
        'proportion_of_exp': 1.,

    },
    params_grid={
        'map_file': [
            'full_mr_map_easy.txt',
        ],
        "lr": [1e-4, 5e-5, 1e-5],
        'rep_size': [64],

        "int_coeff": [0, 1],

        'nepochs': [16],
        'idx': [0, 1, 2],

        # 'lam': [0.95, 0.99],
        # 'gamma': [0.99, 0.999],
        # 'gamma_ext': [0.999, 0.9999],
    },
    script='python3 -m run_toy_mr --mrunner --output_dir=./out --config_file=configs/empty.gin',
    exclude=['.pytest_cache', '.vagrant', '__pycache__',
             'checkpoints', 'out', 'Vagrantfile', 'singularity.def',
             'rnd_toyMR_20200417.simg'],
    python_path='',
    tags=[globals()['script'][:-3], 'rnd', 'easy', '22_09_20', 'eagle', 'final'],
    with_neptune=True
)