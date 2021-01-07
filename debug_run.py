
import warnings
from KFT.job_utils import run_job_func

lags = list(range(1, 25)) + list(range(7 * 24, 8 * 24))
lags_2 = [0,1,2,]
print(lags)
# PATH = ['public_data/' ,'public_movielens_data/' ,'tensor_data/' ,'CCDS_data/' ,'eletric_data/' ,'traffic_data/']
PATH = ['public_data_t_fixed/' ,'public_movielens_data_t_fixed/' ,'tensor_data_t_fixed/'  ,'electric_data/' ,'CCDS_data/','traffic_data/']
dataset = 0
base_dict = {
    'PATH': PATH[dataset],
    'reg_para_a': 0,
    'reg_para_b': 0,
    'batch_size_a': 1e-3,
    'batch_size_b': 1e-2,
    'hyperits': 1,
    'save_path': 'test_run',
    'architecture': 0,
    'task': 'regression',
    'epochs': 50,
    'data_path': PATH[dataset]+'all_data.pt',
    'cuda': True,
    'max_R': 50,
    'max_lr': 1e-2,
    'old_setup': False,
    'latent_scale': False,
    'dual': True,
    'init_max': 1e-1,
    'bayesian': False,
    'multivariate': False,
    'mu_a': 0,
    'mu_b': 0,
    'sigma_a': -1.01,
    'sigma_b': -1.,
    'split_mode': 0,
    'seed': 1,
    'temporal_tag': 2,
    'delete_side_info':None,#None,
    'special_mode': 0,
    'shape_permutation': [0,1,2],
    'full_grad': False,
    'forecast':False,
    'lags':lags,
    'base_ref_int':lags[-1]+1,
    'lambda_W_a':2.,
    'lambda_W_b':2.1,
    'lambda_T_x_a': 100.,#625., for none kernel approach  TRAFFIC: 100-625, CCDS: 500 - 1000
    'lambda_T_x_b': 100.1,#625.1, Try lower values actually for KFT!
    'normalize_Y':False,
    'patience': 100,
    'periods':7,
    'period_size':24,
    'validation_per_epoch': 3,
    'validation_patience': 2,
}
#Do some hyperparameter optimization for kernels...
#Confirming both methods have high potential...
#Fix the transition and convergence rate...
if __name__ == '__main__':
    warnings.simplefilter("ignore")
    run_job_func(base_dict)

