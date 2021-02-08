import os
import warnings
from KFT.job_utils import run_job_func,job_object
import torch
import pickle
PATH = ['public_data/' ,'public_movielens_data_t_fixed/' ,'tensor_data_t_fixed/'  ,'electric_data/' ,'CCDS_data/','traffic_data/']
shape_permutation = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1],
                     [0, 1]]  # Remove this swap this for dimension order
temporal_tag = [2, 2, 2, 0, 2, 0]  # First find the temporal dim mark it if not None
dataset = 0
lags = list(range(0, 25)) + list(range(7 * 24, 8 * 24)) if dataset in [3,5] else [i for i in range(12)]
base_dict = {
    'PATH': PATH[dataset],
    'reg_para_a':0., #for VI dont screw this up
    'reg_para_b': 0., #regularization sets all params to 0? Does not work, figure out why...
    'batch_size_a': 1e-3*8, #8e-3, #Batch size controls "iterations FYI, so might wanna keep this around 100 its"...
    'batch_size_b': 1e-2*1.1,#1.1e-2,
    'hyperits': 5,
    'save_path': 'placeholder',
    'architecture': 0,
    'task': 'regression',
    'epochs': 100,
    'data_path': PATH[dataset]+'all_data.pt',
    'cuda': True,
    'max_R': 50,
    'max_lr': 1e-2,
    'old_setup': False, #Doesnt seem to "train" properly when adding extra terms...
    'latent_scale': False,
    'dual': False,
    'init_max': 1e-1, #fixes regularization issues...
    'bayesian': False,
    'multivariate': True,
    'mu_a': 0,
    'mu_b': 0,
    'sigma_a': -1.01,
    'sigma_b': -1.,
    'split_mode': 0,
    'seed': 1,
    'temporal_tag': 2,
    'delete_side_info':None,#"[1,2],#[0],
    'special_mode': 0,
    'shape_permutation': [0,1,2],#[0,1],
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
    'patience': 50,#100,
    'periods':7,#7, 1
    'period_size':24, #24,15
    'train_core_separate':True,
    'temporal_folds': [0], #Fits well, but does not transfer "back",
    'log_errors':True,
}

def load_best_model(j_tmp,path):
    trials = pickle.load(open(path + f'/frequentist_0.p', "rb"))
    best_ind = sorted(trials.trials, key=lambda x: x['result']['loss'], reverse=True)[0]['misc']['tid'] + 1
    model_info = torch.load(f'{path}/frequentist_0_model_hyperit={best_ind + 1}.pt')
    j_tmp.init(model_info['parameters'])
    j_tmp.load_dumped_model(best_ind + 1)

if __name__ == '__main__':
    ##########FIX PRIMAL BUGGIE

    warnings.simplefilter("ignore")

    if not os.path.exists('WLR_primal_example'):
        base_dict['latent_scale'] = False
        base_dict['save_path'] = 'WLR_primal_example'
        run_job_func(base_dict)
    if not os.path.exists('LS_primal_example'):
        base_dict['latent_scale'] = True
        base_dict['save_path'] = 'LS_primal_example'
        run_job_func(base_dict)

    # else:
    #     j_tmp = job_object(base_dict)
    #     load_best_model(j_tmp,path)



#if no saved model
#train a primal model (small one)
# save this model
#load saved model
#do prediction on test set
# explain these predictions by generalizing which feature is generally important


