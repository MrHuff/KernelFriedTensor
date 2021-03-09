
import warnings
from KFT.job_utils import run_job_func

# PATH = ['public_data/' ,'public_movielens_data/' ,'tensor_data/' ,'CCDS_data/' ,'eletric_data/' ,'traffic_data/']
PATH = ['public_data_t_fixed/' ,'public_movielens_data_t_fixed/' ,'tensor_data_t_fixed/'  ,'electric_data/' ,'CCDS_data/','traffic_data/']
shape_permutation = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1],
                     [0, 1]]  # Remove this swap this for dimension order
temporal_tag = [2, 2, 2, 0, 2, 0]  # First find the temporal dim mark it if not None
dataset = 2
lags = list(range(0, 25)) + list(range(7 * 24, 8 * 24)) if dataset in [3,5] else [i for i in range(12)]
print(lags)
#stuck on train loss for CCDs data Not converging for some reason wtf...
if __name__ == '__main__': #forecasts tends to converge to constant value for some reason...
    warnings.simplefilter("ignore") #memory issues for some reason as well. Likelihood???
    base_dict = {
        'PATH': PATH[dataset],
        'reg_para_a':10., #for VI dont screw this up #Seems that it becomes overregularized in the bayesian case... But what's causing it?!
        'reg_para_b': 100.1, #regularization sets all params to 0? Does not work, figure out why...
        'batch_size_a': 1e-3*8, #8e-3, #Batch size controls "iterations FYI, so might wanna keep this around 100 its"...
        'batch_size_b': 1e-2*1.1,#1.1e-2,
        'hyperits': 1,
        'save_path': 'bayesian_test_run',
        'architecture': 0,
        'task': 'regression',
        'epochs': 20,
        'data_path': PATH[dataset]+'all_data.pt',
        'cuda': True,
        'max_R': 15,
        'max_lr': 1e-2,
        'old_setup': False, #Doesnt seem to "train" properly when adding extra terms...
        'latent_scale': False,
        'dual': True,
        'init_max': 1e0, #fixes regularization issues... Trick for succesfull VI
        'bayesian': True,
        'multivariate': False,
        'mu_a': 0,
        'mu_b': 0,
        'sigma_a': -1,
        'sigma_b': 0,
        'split_mode': 0,
        'seed': 1,
        'temporal_tag': 0 if dataset in [3,5] else 2,
        'delete_side_info': [0,1],#"[1,2],#[0],
        'special_mode': 0,
        'shape_permutation': [0,1] if dataset in [3,5] else [0,2,1],#[0,1],
        'full_grad': False,
        'normalize_Y': False,
        'validation_per_epoch': 5,
        'validation_patience': 2,
        'forecast':False,
        'lags':lags,
        'base_ref_int':lags[-1]+1,
        'lambda_W_a':1.0,
        'lambda_W_b':1.1, #might need to adjust this. CCDS requires higher lambda reg...
        'lambda_T_x_a':0.01,#625., for none kernel approach  TRAFFIC: 100-625, CCDS: 500 - 1000
        'lambda_T_x_b': 0.011,#625.1, Try lower values actually for KFT! #Regularization term seems to blow up if "overtrained on entire set"
        'patience': 500,#100,
        'periods':5 if dataset in [4] else 7,#7, 1
        'period_size':15 if dataset==4 else 24,
        'train_core_separate':True,
        'temporal_folds': [2], #Fits well, but does not transfer "back",
        'log_errors':False
    }
    run_job_func(base_dict)
    # Try to improve convergence somehow..., ok is probably the KL not scaling properly...
    #KL term "takes over optimization" (dont sum the kl when you only optimize fewer stuff...)
    # Not really bug seems to be deeper than that, the frequentist model is also misbehaving..
    # Probably look over autoregressive terms...