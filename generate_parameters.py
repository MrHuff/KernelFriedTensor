import os
import shutil
import pickle

def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

def generate_job_params(directory='job_dir/'):
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
    # PATH = ['public_data/','public_movielens_data/','tensor_data/','CCDS_data/','eletric_data/','traffic_data/']
    PATH = ['public_data_t_fixed/', 'public_movielens_data_t_fixed/', 'tensor_data_t_fixed/','electric_data/',
            'CCDS_data/',
            'traffic_data/']
    data_path = [dataset+ 'all_data.pt' for dataset in PATH]
    seed = [1337]*len(PATH)
    batch_size_a = [1e-3]*len(PATH)
    batch_size_b = [1e-2]*len(PATH)
    reg_para_a = [0]*len(PATH)
    reg_para_b = [0]*len(PATH)
    max_lr = [1e-2]*len(PATH)
    max_R = [200,12,5,100,200,150]
    architecture = [0,0,0,0,0,0]
    temporal_tag = [2,2,2,0,2,0] #First find the temporal dim mark it if not None
    delete_side_info = [None]*len(PATH) #Remove side info, i.e. set to no side info
    special_mode = [0]*len(PATH)
    split_mode = [0]*len(PATH)
    latent_scale = [False]*len(PATH)
    old_setup = [False]*len(PATH)
    cuda = [True]*len(PATH)
    hyperits = [20,20,20,10,20,10]
    epochs = [10]*len(PATH)
    full_grad = [False]*len(PATH)
    dual = [True]*len(PATH)
    sub_epoch_V = [100]*len(PATH)
    chunks = [10]*len(PATH)
    bayesian = [False]*len(PATH)
    task = ['regression']*len(PATH)
    init_max = [1e-1]*len(PATH)
    shape_permutation = [[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1],[0,1]] #Remove this swap this for dimension order
    patience = [50]*len(PATH)
    forecast = [False,False,False,True,False,True]
    lags = [lags_base]*len(PATH)
    base_ref_int=[lags_base[-1]+1]*len(PATH)
    lambda_W_a=[0.5]*len(PATH)
    lambda_W_b=[2.5]*len(PATH)
    lambda_T_x_a=[50]*len(PATH)
    lambda_T_x_b=[500]*len(PATH)
    normalize_Y = [False,False,False,True,True,True]
    periods=[7,7,7,7,1,7]
    period_size=[24,24,24,24,16,24]
    """
    BAYESIAN PARAMS
    """
    multivariate = [False]*len(PATH)
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
        'full_grad':0,
        'sub_epoch_V':0,
        'forecast':True,
        'lags':0,
        'base_ref_int':0,
        'lambda_W_a':2.0,
        'lambda_W_b':2.1,
        'lambda_T_x_a': 150.,#625., for none kernel approach
        'lambda_T_x_b': 150.1,#625.1,
        'normalize_Y':True,
        'patience': 100,
        'periods':7,
        'period_size':24,
    }
    counter = 0

    for i,datasets in enumerate([0]):
        for key in base_dict.keys():
            base_dict[key]=locals()[key][datasets]
        for seed in range(5):
            base_dict['seed']=seed
            print(base_dict)
            save_obj(base_dict,f'job_{counter}',directory)
            counter += 1

if __name__ == '__main__':
    generate_job_params(directory='job_dir_frequentist_alcohol_normal_perm/')
