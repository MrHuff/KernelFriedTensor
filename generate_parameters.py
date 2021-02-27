import os
import shutil
import pickle

def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

def generate_job_params(
                        directory='job_dir/',
                        dataset_ind=0,
                        bayesian_flag=False,
                        mv_flag=False,
                        LS_flag=False,
                        dual_flag=True,
                        old_flag=False,
                        del_list=None,
                        core_flag=True,
                        hyperits_count=10,
                        epochs_count=15
                        ):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    lags_base = list(range(1, 25)) + list(range(7 * 24, 8 * 24))

    """
    BE CAREFUL OF WHAT YOU PUT HERE
    """
    #FIXED PARAMS
    PATH = ['public_data_t_fixed/', 'public_movielens_data_t_fixed/', 'tensor_data_t_fixed/']
    data_path = [dataset+ 'all_data.pt' for dataset in PATH]
    seed = [1337]*len(PATH)
    batch_size_a = [1e-3*8]*len(PATH)
    batch_size_b = [1e-2*1.2]*len(PATH)
    reg_para_a = [1e-2]*len(PATH)
    reg_para_b = [100]*len(PATH)
    max_lr = [1e-2]*len(PATH)
    max_R = [30,6,5]
    architecture = [0,0,0] #[0,0,1]
    temporal_tag = [2,2,2] #First find the temporal dim mark it if not None
    delete_side_info = [None]*len(PATH) #Remove side info, i.e. set to no side info
    delete_side_info[dataset_ind]=del_list
    special_mode = [0]*len(PATH)
    split_mode = [0]*len(PATH)
    latent_scale = [LS_flag]*len(PATH)
    old_setup = [old_flag]*len(PATH)
    cuda = [True]*len(PATH)
    hyperits = [hyperits_count]*len(PATH)
    epochs = [epochs_count]*len(PATH)
    full_grad = [False]*len(PATH)
    dual = [dual_flag]*len(PATH)
    task = ['regression']*len(PATH)
    init_max = [1e0]*len(PATH)
    shape_permutation = [[0,1,2],[0,1,2],[0,1,2]] #Remove this swap this for dimension order
    patience = [100]*len(PATH)
    forecast = [False,False,False,True,False,True]
    lags = [[0],[0],[0]]
    base_ref_int=[0,0,0]
    lambda_W_a=[0.5]*len(PATH)
    lambda_W_b=[2.5]*len(PATH)
    lambda_T_x_a=[50]*len(PATH)
    lambda_T_x_b=[500]*len(PATH)
    normalize_Y = [False,False,False,True,True,True]
    periods=[7,7,7]
    period_size=[24,24,24]
    validation_per_epoch=[4,4,4]
    validation_patience = [2,2,2]
    """
    BAYESIAN PARAMS
    """
    bayesian = [bayesian_flag]*len(PATH)
    multivariate = [mv_flag]*len(PATH)
    mu_a = [0.]*len(PATH)
    mu_b =[0.]*len(PATH)
    sigma_a = [-1]*len(PATH)
    sigma_b = [2]*len(PATH)

    base_dict={
        'PATH':0,
        'reg_para_a':0,
        'reg_para_b':0,
        'batch_size_a':0,
        'batch_size_b':0,
        'hyperits':0,
        'architecture':0,
        'task':0,
        'epochs':0,
        'bayesian':0,
        'data_path':0,
        'cuda':0,
        'max_R':0,
        'max_lr':0,
        'old_setup':0,
        'latent_scale':0,
        'chunks':0,
        'dual':0,
        'init_max':0,
        'multivariate':0,
        'mu_a':0,
        'sigma_a':0,
        'mu_b':0,
        'sigma_b':0,
        'split_mode':0,
        'seed':0,
        'temporal_tag':0,
        'delete_side_info':0,
        'special_mode':0,
        'shape_permutation': 0,
        'full_grad':False,
        'forecast':False,
        'lags':0,
        'base_ref_int':0,
        'lambda_W_a':0.0,
        'lambda_W_b':0.5,
        'lambda_T_x_a': 100.,#625., for none kernel approach
        'lambda_T_x_b': 100.1,#625.1,
        'normalize_Y':False,
        'patience': 100,
        'periods':7,
        'period_size':24,
        'validation_per_epoch': 3,
        'validation_patience': 2,
        'train_core_separate': core_flag,
        'temporal_folds': [0],
        'log_errors':False

    }
    counter = 0

    for i,datasets in enumerate([dataset_ind]):
        for key in base_dict.keys():
            if key in locals().keys():
                base_dict[key]=locals()[key][datasets]
        for seed in range(5):
            base_dict['seed']=seed
            print(base_dict)
            save_obj(base_dict,f'job_{counter}',directory)
            counter += 1

if __name__ == '__main__':
    generate_job_params(directory='alchohol_bayesian_benchmark/',
                         dataset_ind=0,
                        bayesian_flag=True,
                        mv_flag=False,
                        LS_flag=False,
                        dual_flag=True,
                        old_flag=True,
                        core_flag=True,
                        del_list=[0,1,2],
                        hyperits_count=10,
                        epochs_count=25
                        )
    generate_job_params(directory='movielens_20_benchmark_bayesian/',
                         dataset_ind=1,
                        bayesian_flag=True,
                        mv_flag=False,
                        LS_flag=False,
                        dual_flag=True,
                        old_flag=True,
                        core_flag=True,
                        del_list=[1,2],
                        hyperits_count=10,
                        epochs_count=15
                        )
    generate_job_params(directory='movielens_20_bayesian_dual_multivariate/',
                         dataset_ind=1,
                        bayesian_flag=True,
                        mv_flag=True,
                        LS_flag=False,
                        dual_flag=True,
                        old_flag=False,
                        del_list=None,
                        core_flag=True,
                        hyperits_count=10,
                        epochs_count=15
                        )
    generate_job_params(directory='movielens_20_bayesian_dual_multivariate_LS/',
                         dataset_ind=1,
                        bayesian_flag=True,
                        mv_flag=True,
                        LS_flag=True,
                        dual_flag=True,
                        old_flag=False,
                        del_list=None,
                        core_flag=True,
                        hyperits_count=10,
                        epochs_count=15
                        )
    generate_job_params(directory='movielens_20_bayesian_dual_univariate/',
                         dataset_ind=1,
                        bayesian_flag=True,
                        mv_flag=False,
                        LS_flag=False,
                        dual_flag=True,
                        old_flag=False,
                        del_list=None,
                        core_flag=True,
                        hyperits_count=10,
                        epochs_count=15
                        )
    generate_job_params(directory='movielens_20_bayesian_dual_univariate_LS/',
                         dataset_ind=1,
                        bayesian_flag=True,
                        mv_flag=False,
                        LS_flag=True,
                        dual_flag=True,
                        old_flag=False,
                        del_list=None,
                        core_flag=True,
                        hyperits_count=10,
                        epochs_count=15
                        )