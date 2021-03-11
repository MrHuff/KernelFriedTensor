from generate_parameters_bayesian import *


def generate_job_params(directory='job_dir/',hyperits=5,LS=False,dual=False,mv=False,bayesian=False,forecast=False,normalize=False,seperate_train=False):
    dataset= 2
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    PATH = ['electric_data/',
            'CCDS_data/',
            'traffic_data/']
    lags_base =  list(range(0, 25)) + list(range(7 * 24, 8 * 24)) if dataset in [0,2] else [i for i in range(12)]
    base_dict={
        'PATH':PATH[dataset],
        'reg_para_a':0,
        'reg_para_b':0,
        'batch_size_a':1e-3*8,
        'batch_size_b':1e-2*1.2,
        'hyperits':hyperits,
        'architecture':0,
        'task':'regression',
        'epochs':50,
        'bayesian':bayesian,
        'data_path':PATH[dataset]+'all_data.pt',
        'cuda':True,
        'max_R':100,
        'max_lr':1e-2,
        'old_setup':False,
        'latent_scale':LS,
        'dual':dual,
        'init_max':1e-1,
        'multivariate':mv,
        'mu_a':0,
        'sigma_a':0,
        'mu_b':0,
        'sigma_b':0,
        'split_mode':0,
        'seed':0,
        'temporal_tag':0 if dataset in [0,2] else 2,
        'delete_side_info':None,
        'special_mode':0,
        'shape_permutation': [0,1] if dataset in [0,2] else [1,0,2],
        'full_grad':False,
        'forecast':forecast,
        'lags':lags_base,
        'base_ref_int':lags_base[-1]+1,
        'lambda_W_a':0.0,
        'lambda_W_b':0.5,
        'lambda_T_x_a': 100.,#625., for none kernel approach
        'lambda_T_x_b': 100.1,#625.1,
        'normalize_Y':normalize,
        'patience': 100,
        'periods':7 if dataset in [0,2] else 5,
        'period_size':12 if dataset in [0,2] else 15,
        'validation_per_epoch': 5,
        'validation_patience': 2,
        'train_core_separate':seperate_train,
        'temporal_folds': [0],
        'log_errors': False
    }
    counter = 0
    rang = 7 if dataset in [0,2] else 5
    for W in [0.0]:
    # for W in [0.0,0.1,1.0,2.0,5.0,10.]:
        for lt in [625.]:
        # for lt in [10.,100.,1000.,10000.]:
            for t in range(rang):
                base_dict['lambda_T_x_a']=lt*0.9
                base_dict['lambda_T_x_b']=lt*1.1
                base_dict['lambda_W_a']=W*0.5
                base_dict['lambda_W_b']=W+1.5
                base_dict['temporal_folds']=[t]
                print(base_dict)
                save_obj(base_dict,f'job_{counter}',directory)
                counter += 1

if __name__ == '__main__':
    generate_job_params('jobs_traffic_side_info_WLR_horizon=12/',
                        hyperits=10, LS=False, dual=True, mv=False, bayesian=False, forecast=True, normalize=True,
                        seperate_train=True
                        )
    generate_job_params('jobs_traffic_side_info_LS_horizon=12/',
                        hyperits=10, LS=True, dual=True, mv=False, bayesian=False, forecast=True, normalize=True,
                        seperate_train=True
                        )
