
import warnings
from KFT.job_utils import run_job_func
import numpy as np
import matplotlib.pyplot as plt
import os
PATH = ['public_data_t_fixed/' ,'public_movielens_data_t_fixed/' ,'tensor_data_t_fixed/'  ,'electric_data/' ,'CCDS_data/','traffic_data/']
shape_permutation = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1],
                     [0, 1]]  # Remove this swap this for dimension order
temporal_tag = [2, 2, 2, 0, 2, 0]  # First find the temporal dim mark it if not None
dataset = 0
lags = list(range(0, 25)) + list(range(7 * 24, 8 * 24)) if dataset in [3,5] else [i for i in range(12)]
#stuck on train loss for CCDs data
base_dict = {
    'PATH': PATH[dataset],
    'reg_para_a':0., #for VI dont screw this up
    'reg_para_b': 1e-2, #regularization sets all params to 0? Does not work, figure out why...
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
    'old_setup': True, #Doesnt seem to "train" properly when adding extra terms...
    'latent_scale': False,
    'dual': True,
    'init_max': 1e-1, #fixes regularization issues...
    'bayesian': False,
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
    'patience': 500000,#100,
    'periods':7,#7, 1
    'period_size':24, #24,15
    'train_core_separate':True,
    'temporal_folds': [0], #Fits well, but does not transfer "back",
    'log_errors': True
}

font_size = 30
plt.rcParams['font.size'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['figure.figsize'] = 15, 7.5
plt.rcParams['axes.titlesize'] = 35
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

def load_and_concat(path,file):
    data_container = []
    min_lengths = np.inf
    for i in range(1,6):
        vec = np.load(path+f'/{file}_{i}.npy')
        data_container.append(vec)
        length = vec.shape[0]
        if length<min_lengths:
            min_lengths = length
    # for i in range(len(data_container)):
    #     data_container[i] = data_container[i][:min_lengths]

    return np.stack(data_container)
def mean_std(vec):
    mean = vec.mean(axis=0)
    std = vec.std(axis=0)
    ci_neg = mean-std
    ci_pos = mean+std
    # ci_neg = vec.min(axis=0)
    # ci_pos = vec.max(axis=0)
    x = np.arange(1,mean.shape[0]+1)
    return mean,ci_neg,ci_pos,x

def plot_point(save_name,errors,title,path,mode):

    path_2 = 'KFT_motivation_old_setup'
    vec = load_and_concat(path,errors)
    # for el in vec:
    #     plt.plot(el,color='b')
    mean,ci_neg,ci_pos,x = mean_std(vec)
    plt.plot(x,mean,label=f'{mode}')
    plt.fill_between(x,ci_neg,ci_pos,color='b',alpha=0.1)
    vec_2 = load_and_concat(path_2, errors)
    # for el in vec_2:c
    #     plt.plot(el,color='r')
    mean, ci_neg, ci_pos, x = mean_std(vec_2)
    plt.plot(x, mean,label='Naive')
    plt.fill_between(x, ci_neg, ci_pos, color='r', alpha=0.1)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{save_name}.png',bbox_inches = 'tight',
    pad_inches = 0.1)
    plt.clf()


if __name__ == '__main__':

    warnings.simplefilter("ignore")
    if not os.path.exists('KFT_motivation'):
        base_dict['save_path'] = 'KFT_motivation'
        base_dict['old_setup'] = False
        run_job_func(base_dict)
    if not os.path.exists('KFT_motivation_LS'):
        base_dict['save_path'] = 'KFT_motivation_LS'
        base_dict['latent_scale']=True
        base_dict['old_setup'] = False
        run_job_func(base_dict)
    if not os.path.exists('KFT_motivation_old_setup'):
        base_dict['save_path'] = 'KFT_motivation_old_setup'
        base_dict['old_setup'] = True
        run_job_func(base_dict)

    plot_point('train','train_errors','Training error',path = 'KFT_motivation',mode='KFT-WLR')
    plot_point('val','val_errors','Validation error',path = 'KFT_motivation',mode='KFT-WLR')
    plot_point('test','test_errors','Test error',path = 'KFT_motivation',mode='KFT-WLR')

    plot_point('train_LS','train_errors','Training error',path = 'KFT_motivation_LS',mode='KFT-LS')
    plot_point('val_LS','val_errors','Validation error',path = 'KFT_motivation_LS',mode='KFT-LS')
    plot_point('test_LS','test_errors','Test error',path = 'KFT_motivation_LS',mode='KFT-LS')


