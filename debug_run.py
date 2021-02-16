
import warnings
from KFT.job_utils import run_job_func

# PATH = ['public_data/' ,'public_movielens_data/' ,'tensor_data/' ,'CCDS_data/' ,'eletric_data/' ,'traffic_data/']
PATH = ['public_data_t_fixed/' ,'public_movielens_data_t_fixed/' ,'tensor_data_t_fixed/'  ,'electric_data/' ,'CCDS_data/','traffic_data/']
shape_permutation = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1],
                     [0, 1]]  # Remove this swap this for dimension order
temporal_tag = [2, 2, 2, 0, 2, 0]  # First find the temporal dim mark it if not None
dataset = 0
lags = list(range(0, 25)) + list(range(7 * 24, 8 * 24)) if dataset in [3,5] else [i for i in range(12)]
print(lags)
#stuck on train loss for CCDs data Not converging for some reason wtf...
base_dict = {
    'PATH': PATH[dataset],
    'reg_para_a':1e-2, #for VI dont screw this up
    'reg_para_b': 10., #regularization sets all params to 0? Does not work, figure out why...
    'batch_size_a': 1e-3*8, #8e-3, #Batch size controls "iterations FYI, so might wanna keep this around 100 its"...
    'batch_size_b': 1e-2*1.1,#1.1e-2,
    'hyperits': 2,
    'save_path': 'bayesian_test_run',
    'architecture': 0,
    'task': 'regression',
    'epochs': 2,
    'data_path': PATH[dataset]+'all_data.pt',
    'cuda': True,
    'max_R': 20,
    'max_lr': 1e-2,
    'old_setup': False, #Doesnt seem to "train" properly when adding extra terms...
    'latent_scale': False,
    'dual': True,
    'init_max': 1e0, #fixes regularization issues... Trick for succesfull VI
    'bayesian': True,
    'multivariate': False,
    'mu_a': 0,
    'mu_b': 0,
    'sigma_a': -1.01,
    'sigma_b': -1.,
    'split_mode': 0,
    'seed': 1,
    'temporal_tag': 2,
    'delete_side_info':None,#"[1,2],#[0],
    'special_mode': 0,
    'shape_permutation': [0,2,1],#[0,1],
    'full_grad': False,
    'normalize_Y': False,
    'validation_per_epoch': 5,
    'validation_patience': 2,
    'forecast':False,
    'lags':lags,
    'base_ref_int':lags[-1]+1,
    'lambda_W_a':0.,
    'lambda_W_b':0.+1e-4, #might need to adjust this. CCDS requires higher lambda reg...
    'lambda_T_x_a': 10000.,#625., for none kernel approach  TRAFFIC: 100-625, CCDS: 500 - 1000
    'lambda_T_x_b': 10000+1e-4,#625.1, Try lower values actually for KFT! #Regularization term seems to blow up if "overtrained on entire set"
    'patience': 500,#100,
    'periods':7,#7, 1
    'period_size':24, #24,15
    'train_core_separate':True,
    'temporal_folds': [0], #Fits well, but does not transfer "back",
    'log_errors':False
}
# base_dict={'PATH': 'public_data_t_fixed/', 'reg_para_a': 0.01, 'reg_para_b': 100, 'batch_size_a': 0.008, 'batch_size_b': 0.012, 'hyperits': 10, 'architecture': 0, 'task': 'regression', 'epochs': 50, 'bayesian': True, 'data_path': 'public_data_t_fixed/all_data.pt', 'cuda': True, 'max_R': 20, 'max_lr': 0.01, 'old_setup': False, 'latent_scale': False, 'chunks': 0, 'dual': True, 'init_max': 1.0, 'multivariate': False, 'mu_a': 0.0, 'sigma_a': -1, 'mu_b': 0.0, 'sigma_b': 2, 'split_mode': 0, 'seed': 0, 'temporal_tag': 2, 'delete_side_info': None, 'special_mode': 0, 'shape_permutation': [0, 1, 2], 'full_grad': False, 'forecast': False, 'lags': [0], 'base_ref_int': 0, 'lambda_W_a': 0.5, 'lambda_W_b': 2.5, 'lambda_T_x_a': 50, 'lambda_T_x_b': 500, 'normalize_Y': False, 'patience': 100, 'periods': 7, 'period_size': 24, 'validation_per_epoch': 4, 'validation_patience': 2, 'train_core_separate': True, 'temporal_folds': [0], 'log_errors': False}
# base_dict['save_path'] = 'debug'
#Do some hyperparameter optimization for kernels...
#Confirming both methods have high potential...
#Fix the transition and convergence rate...
if __name__ == '__main__':
    warnings.simplefilter("ignore") #memory issues for some reason as well. Likelihood???
    run_job_func(base_dict)

