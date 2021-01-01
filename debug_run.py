
import warnings
from KFT.job_utils import run_job_func

lags = list(range( 24)) + list(range(7 * 24, 8 * 24))

# PATH = ['public_data/' ,'public_movielens_data/' ,'tensor_data/' ,'CCDS_data/' ,'eletric_data/' ,'traffic_data/']
PATH = ['public_data_t_fixed/' ,'public_movielens_data_t_fixed/' ,'tensor_data_t_fixed/' ,'CCDS_data/' ,'electric_data/' ,'traffic_data/']
dataset = -1
base_dict = {
    'PATH': PATH[dataset],
    'reg_para_a': 0,
    'reg_para_b': 0,
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
    'old_setup': True,
    'latent_scale': False,
    'chunks': 10,
    'dual': False,
    'init_max': 1e-1,
    'bayesian': False,
    'multivariate': False,
    'mu_a': 0,
    'mu_b': 0,
    'sigma_a': -1.01,
    'sigma_b': -1.,
    'split_mode': 0,
    'seed': 1,
    'temporal_tag': 0,
    'delete_side_info':[0],
    'special_mode': 0,
    'shape_permutation': [0,1],
    'full_grad': False,
    'sub_epoch_V': 100,
    'forecast':True,
    'lags':lags,
    'base_ref_int':lags[-1]+24,
    'lambda_W_a':10.0,
    'lambda_W_b':10.1,
    'lambda_T_x_a': 625.,
    'lambda_T_x_b': 625.1,
}
if __name__ == '__main__':
    warnings.simplefilter("ignore")
    run_job_func(base_dict)
