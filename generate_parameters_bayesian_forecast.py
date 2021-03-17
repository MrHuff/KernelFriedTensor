from generate_parameters_bayesian import *


def generate_job_params(dataset=2,directory='job_dir/',hyperits=5,LS=False,dual=False,
                        mv=False,bayesian=False,forecast=False,normalize=False,
                        seperate_train=False,a=1.0,b=100,max_R = 75,init_val=1e-1):
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
        'reg_para_a':a,
        'reg_para_b':b,
        'batch_size_a':1e-3*8,
        'batch_size_b':1e-2*1.2,
        'hyperits':hyperits,
        'architecture':0,
        'task':'regression',
        'epochs':25,
        'bayesian':bayesian,
        'data_path':PATH[dataset]+'all_data.pt',
        'cuda':True,
        'max_R':max_R,
        'max_lr':1e-2,
        'old_setup':False,
        'latent_scale':LS,
        'dual':dual,
        'init_max':init_val,
        'multivariate':mv,
        'mu_a':1e-2,
        'sigma_a':-1,
        'mu_b':0.5,
        'sigma_b':0.5,
        'split_mode':0,
        'seed':0,
        'temporal_tag':0 if dataset in [0,2] else 2,
        'delete_side_info':None,
        'special_mode':0,
        'shape_permutation': [0,1] if dataset in [0,2] else [0,1,2],
        'full_grad':False,
        'forecast':forecast,
        'lags':lags_base,
        'base_ref_int':lags_base[-1]+1,
        'lambda_W_a':0.5,
        'lambda_W_b':2.0,
        'lambda_T_x_a': 1e-2,#625., for none kernel approach
        'lambda_T_x_b': 1e-1,#625.1,
        'normalize_Y':normalize,
        'patience': 100,
        'periods':3,
        'period_size':24 if dataset in [0,2] else 15,
        'validation_per_epoch': 5,
        'validation_patience': 2,
        'train_core_separate':seperate_train,
        'temporal_folds': [0],
        'log_errors': False
    }
    counter = 0
    rang = 3
    for W in [1e-2]:
    # for W in [0.0,0.1,1.0,2.0,5.0,10.]:
        for lt in [1e-2]:
        # for lt in [10.,100.,1000.,10000.]:
            for t in range(rang):
                base_dict['lambda_T_x_a']=lt*1e-1
                base_dict['lambda_T_x_b']=lt*10
                base_dict['lambda_W_a']=W*1e-1
                base_dict['lambda_W_b']=W*10
                base_dict['temporal_folds']=[t]
                print(base_dict)
                save_obj(base_dict,f'job_{counter}',directory)
                counter += 1

if __name__ == '__main__':
    generate_job_params(dataset=2,directory='jobs_traffic_baysian_WLR_3/',
                        hyperits=5, LS=False, dual=True, mv=False, bayesian=True, forecast=True, normalize=True,
                        seperate_train=True,a=1e6,b=2e6,init_val=1e-2
                        )
    generate_job_params(dataset=2,directory='jobs_traffic_baysian_LS_3/',
                        hyperits=5, LS=True, dual=True, mv=False, bayesian=True, forecast=True, normalize=True,
                        seperate_train=True,a=1e6,b=2e6,init_val=1e-2
                        )

    generate_job_params(dataset=1,directory='jobs_CCDS_baysian_WLR_3/',
                        hyperits=10, LS=False, dual=True, mv=False, bayesian=True, forecast=True, normalize=True,
                        seperate_train=True,a=1,b=1e6,max_R=100,init_val=1e-1
                        )
    generate_job_params(dataset=1,directory='jobs_CCDS_baysian_LS_3/',
                        hyperits=10, LS=True, dual=True, mv=False, bayesian=True, forecast=True, normalize=True,
                        seperate_train=True,a=1,b=100,max_R=100,init_val=1e-1
                        )
