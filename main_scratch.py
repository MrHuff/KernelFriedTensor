from pykeops.torch import KernelSolve
from KFT.util import process_old_setup,load_side_info,concat_old_side_info
from KFT.job_utils import job_object
import pickle
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)



def Kinv_keops(x, b, gamma, alpha):
    N=10000
    D=5
    Dv=2
    formula = 'Exp(- g * SqDist(x,y)) * a'
    aliases = ['x = Vi(' + str(D) + ')',  # First arg:  i-variable of size D
               'y = Vj(' + str(D) + ')',  # Second arg: j-variable of size D
               'a = Vj(' + str(Dv) + ')',  # Third arg:  j-variable of size Dv
               'g = Pm(1)']  # Fourth arg: scalar parameter
    Kinv = KernelSolve(formula, aliases, "a", axis=1)
    res = Kinv(x, x, b, gamma, alpha=alpha)
    return res

if __name__ == '__main__':
    with warnings.catch_warnings(): #There are some autograd issues fyi, might wanna fix it sooner or later
        warnings.simplefilter("ignore")
        PATH = './experiment_3/'
        # process_old_setup(PATH,'tensor_data.pt')

        # concat_old_side_info(PATH,['location_tensor_400000.pt','article_tensor_400000.pt','time_tensor_400000.pt'])
        side_info = load_side_info(side_info_path=PATH,indices=[1,0,2])
        print(side_info[0]['data'].shape)
        shape = pickle.load(open(PATH+'full_tensor_shape.pickle','rb'))
        side_info[2]['temporal'] = True
        tensor_architecture = {0:{'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':10,'has_side_info':True,},
                               1: {'ii': [1], 'r_1': 10, 'n_list': [shape[1]], 'r_2': 10, 'has_side_info': True, },
                               2: {'ii': [2], 'r_1': 10, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True, },
                               }
        other_configs={
            'reg_para_a':1e-6,
            'reg_para_b': 1e-2,
            'batch_size_a': 1.0,
            'batch_size_b': 1.0,
            'fp_16':False,
            'fused':False,
            'hyperits':2,
            'save_path': './private_data_test/',
            'job_name':'frequentist',
            'task':'reg',
            'epochs': 10,
            'bayesian': False,
            'data_path':PATH+'all_data.pt',
            'cuda':True,
            'device':'cuda:0',
            'train_loss_interval_print':10,
            'sub_epoch_V':100,
            'sub_epoch_ls':100,
            'config':{'full_grad':True}
        }
        j = job_object(
            side_info_dict=side_info,
            tensor_architecture=tensor_architecture,
            other_configs=other_configs,
            seed=1
        )
        j.run_hyperparam_opt()

