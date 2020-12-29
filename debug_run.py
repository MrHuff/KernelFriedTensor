
import warnings
from KFT.job_utils import run_job_func


# PATH = ['public_data/' ,'public_movielens_data/' ,'tensor_data/' ,'CCDS_data/' ,'eletric_data/' ,'traffic_data/']
PATH = ['public_data_t_fixed/' ,'public_movielens_data_t_fixed/' ,'tensor_data_t_fixed/' ,'CCDS_data/' ,'electric_data/' ,'traffic_data/']
dataset = 0
base_dict = {
    'PATH': PATH[dataset],
    'reg_para_a': 2e0,
    'reg_para_b': 2e0+1e-3,
    'batch_size_a': 1e-2,
    'batch_size_b': 5e-2,
    'hyperits': 1,
    'save_path': 'test_run',
    'architecture': 0,
    'task': 'regression',
    'epochs': 10,
    'data_path': PATH[dataset]+'all_data.pt',
    'cuda': True,
    'max_R': 25,
    'max_lr': 1e-1,
    'old_setup': False,
    'latent_scale': False,
    'chunks': 10,
    'dual': True,
    'init_max': 1e0,
    'bayesian': True,
    'multivariate': False,
    'mu_a': 0,
    'mu_b': 0,
    'sigma_a': -2,
    'sigma_b': -1,
    'split_mode': 0,
    'seed': 1,
    'temporal_tag': 2,
    'delete_side_info':None,
    'special_mode': 0,
    'shape_permutation': [0,1,2],
    'full_grad': False,
    'sub_epoch_V': 100,
}

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    run_job_func(base_dict)
