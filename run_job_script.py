from KFT.util import process_old_setup,load_side_info,concat_old_side_info
from KFT.job_utils import job_object
import pickle
import warnings
from KFT.util import job_parser,get_free_gpu
import os
def get_tensor_architectures(i,shape):
    TENSOR_ARCHITECTURES = {
        0:{
           0:{'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':10,'has_side_info':True},
           1: {'ii': [1], 'r_1': 10, 'n_list': [shape[1]], 'r_2': 10, 'has_side_info': True}, #Magnitude of kernel sum
           2: {'ii': [2], 'r_1': 10, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True},
           },
        1:{
           0:{'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':10,'has_side_info':True},
           1: {'ii': [1,2], 'r_1': 10, 'n_list': [shape[1],shape[2]], 'r_2': 1, 'has_side_info': True}, #Magnitude of kernel sum
           },
        2: {
            0: {'ii': [0,1], 'r_1': 1, 'n_list': [shape[0],shape[1]], 'r_2': 10, 'has_side_info': True},
            1: {'ii': [2], 'r_1': 10, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True},
        },
    }
    return TENSOR_ARCHITECTURES[i]

if __name__ == '__main__':
    with warnings.catch_warnings(): #There are some autograd issues fyi, might wanna fix it sooner or later
        warnings.simplefilter("ignore")
        gpu = get_free_gpu(8)
        args = vars(job_parser())
        print(args)
        PATH = args['PATH']
        if not os.path.exists(PATH+'all_data.pt'):
            process_old_setup(PATH,tensor_name=args['tensor_name'])
            concat_old_side_info(PATH,args['side_info_name'])

        side_info = load_side_info(side_info_path=PATH,indices=args['side_info_order'])
        shape = pickle.load(open(PATH+'full_tensor_shape.pickle','rb'))
        for i in args['temporal_tag']:
            side_info[i]['temporal'] = True

        tensor_architecture = get_tensor_architectures(args['architecture'],shape)
        other_configs={
            'reg_para_a':args['reg_para_a'], #Regularization term! Need to choose wisely
            'reg_para_b': args['reg_para_b'],
            'batch_size_a': 1.0 if args['full_grad'] else args['batch_size_a'],
            'batch_size_b': 1.0 if args['full_grad'] else args['batch_size_b'],
            'fp_16': args['fp_16'], #Wanna use fp_16? Initialize smartly!
            'fused': args['fused'],
            'hyperits':args['hyperits'],
            'save_path': args['save_path'],
            'task':args['task'],
            'epochs': args['epochs'],
            'bayesian': args['bayesian'], #Mean field does not converge to something meaningful?!
            'data_path':PATH+'all_data.pt',
            'cuda':args['cuda'],
            'device':f'cuda:{gpu[0]}',
            'train_loss_interval_print':args['sub_epoch_V']//10,
            'sub_epoch_V':args['sub_epoch_V'],
            'sub_epoch_ls':args['sub_epoch_ls'],
            'sub_epoch_prime': args['sub_epoch_prime'],
            'config':{'full_grad':args['full_grad']}
        }
        j = job_object(
            side_info_dict=side_info,
            tensor_architecture=tensor_architecture,
            other_configs=other_configs,
            seed=args['seed']
        )
        j.run_hyperparam_opt()
