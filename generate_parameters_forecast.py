from generate_parameters import *


def generate_job_params(directory='job_dir/'):
    dataset= 1
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
        'hyperits':15,
        'architecture':0,
        'task':'regression',
        'epochs':200,
        'bayesian':False,
        'data_path':PATH[dataset]+'all_data.pt',
        'cuda':True,
        'max_R':200,
        'max_lr':1e-2,
        'old_setup':False,
        'latent_scale':False,
        'dual':True,
        'init_max':1e-1,
        'multivariate':False,
        'mu_a':0,
        'sigma_a':0,
        'mu_b':0,
        'sigma_b':0,
        'split_mode':0,
        'seed':0,
        'temporal_tag':0 if dataset in [0,2] else 2,
        'delete_side_info':[1,2],
        'special_mode':0,
        'shape_permutation': [0,1] if dataset in [0,2] else [1,0,2],
        'full_grad':False,
        'forecast':True,
        'lags':lags_base,
        'base_ref_int':lags_base[-1]+1,
        'lambda_W_a':0.0,
        'lambda_W_b':0.5,
        'lambda_T_x_a': 100.,#625., for none kernel approach
        'lambda_T_x_b': 100.1,#625.1,
        'normalize_Y':False,
        'patience': 200,
        'periods':7 if dataset in [0,2] else 5,
        'period_size':24 if dataset in [0,2] else 15,
        'validation_per_epoch': 5,
        'validation_patience': 2,
        'train_core_separate':True,
        'temporal_folds': [0]
    }
    counter = 0
    rang = 7 if dataset in [0,2] else 5
    for W in [2.]:
    # for W in [0.0,0.1,1.0,2.0,5.0,10.]:
        for lt in [10000.]:
        # for lt in [10.,100.,1000.,10000.]:
            for t in range(rang):
                base_dict['lambda_T_x_a']=lt*0.5
                base_dict['lambda_T_x_b']=lt*1.5
                base_dict['lambda_W_a']=W
                base_dict['lambda_W_b']=W+10
                base_dict['temporal_folds']=[t]
                print(base_dict)
                save_obj(base_dict,f'job_{counter}',directory)
                counter += 1

if __name__ == '__main__':
    generate_job_params('jobs_CCDS_4/')