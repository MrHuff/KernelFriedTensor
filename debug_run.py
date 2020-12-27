
import warnings
from KFT.job_utils import run_job_func


# PATH = ['public_data/' ,'public_movielens_data/' ,'tensor_data/' ,'CCDS_data/' ,'eletric_data/' ,'traffic_data/']
PATH = ['public_data_t_fixed/' ,'public_movielens_data_t_fixed/' ,'tensor_data_t_fixed/' ,'CCDS_data/' ,'electric_data/' ,'traffic_data/']
dataset = 5
base_dict = {
    'PATH': PATH[dataset],
    'reg_para_a': 1e-5,
    'reg_para_b': 1e-4,
    'batch_size_a': 1e-2,
    'batch_size_b': 1e-1,
    'hyperits': 1,
    'save_path': 'test_run',
    'architecture': 0,
    'task': 'regression',
    'epochs': 400,
    'data_path': PATH[dataset]+'all_data.pt',
    'cuda': True,
    'max_R': 50,
    'max_lr': 1e-2,
    'old_setup': False,
    'latent_scale': False,
    'chunks': 10,
    'dual': True,
    'init_max': 1e-1,
    'bayesian': False,
    'multivariate': False,
    'mu_a': 0,
    'sigma_a': -1,
    'mu_b': 0,
    'sigma_b': 0,
    'split_mode': 2,
    'seed': 1337,
    'temporal_tag': [0],
    'delete_side_info':None,
    'special_mode': 0,
    'shape_permutation': [0,1],
    'full_grad': False,
    'sub_epoch_V': 100,
}

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    run_job_func(base_dict)
