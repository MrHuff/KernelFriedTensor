from pykeops.torch import KernelSolve
from KFT.util import process_old_setup,load_side_info
from KFT.job_utils import job_object
import pickle
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

    PATH = './experiment_3/'
    side_info = load_side_info(side_info_path=PATH,indices=[1,0,2])

    shape = pickle.load(open(PATH+'full_tensor_shape.pickle','rb'))
    side_info[2]['temporal'] = True
    tensor_architecture = {0:{'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':10,'has_side_info':True,},
                           1: {'ii': [1], 'r_1': 10, 'n_list': [shape[1]], 'r_2': 10, 'has_side_info': True, },
                           2: {'ii': [2], 'r_1': 10, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True, },
                           }
    other_configs={
        'reg_para_a':1e-6,
        'reg_para_b': 1e-2,
        'batch_size_a':1e-3,
        'batch_size_b': 1.0,
        'fp_16':False,
        'fused':False,
        'hyperits':2,
        'save_path': './test_data_job/',
        'job_name':'frequentist',
        'task':'reg',
        'epochs': 100,
        'bayesian': False,
        'data_path':PATH+'all_data.pt',
        'cuda':True,
        'device':'cuda:0',
        'train_loss_interval_print':10,
        'sub_epoch_V':5,
        'sub_epoch_ls':5,
    }
    j = job_object(
        side_info_dict=side_info,
        tensor_architecture=tensor_architecture,
        other_configs=other_configs,
        seed=1
    )
    j.run_hyperparam_opt()

    #     #Doesn't like last index being 1 -> do squeeze type operation!
    # o = kernel_adding_tensor(PATH)
    # print(o.data.shape[0])
    # print(o.data.shape[1])
    # print(o.data.shape[2])
    # cuda_device = 'cuda:0'
    # init_dict = {
    #             # 0: {'ii': [0, 1], 'lambda': 1e-3, 'r_1': 1, 'n_list': [o.data.shape[0], o.data.shape[1]], 'r_2': 10,'has_side_info': True, 'side_info': {1:o.n_side,2:o.m_side},'kernel_para': {'ls_factor': 1.0, 'kernel_type': 'rbf', 'nu': 2.5}},
    #             0:{'ii':[0],'r_1':1,'n_list':[o.data.shape[0]],'r_2':10,'has_side_info':True,'side_info':{1:o.n_side.to(cuda_device)},'kernel_para':{1:{'ARD':True,'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5,}} },
    #             # 1:{'ii':[2],'lambda':1e-3,'r_1':10,'n_list':[o.data.shape[2]],'r_2':1,'has_side_info':True,'side_info':{1:o.t_side},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5} },
    #             1:{'ii':[1,2],'r_1':10,'n_list':[o.data.shape[1],o.data.shape[2]],'r_2':1,'has_side_info':True,'side_info':{1:o.m_side.to(cuda_device),2:o.t_side.to(cuda_device)},'kernel_para':{1:{'ARD':False,'ls_factor':1.0, 'kernel_type':'matern','nu':2.5,},2:{'ARD':False,'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5,'deep':False}}},
    #             # 1:{'ii':1,'lambda':0.0001,'r_1':10,'n_list':[o.data.shape[1]],'r_2':10,'has_side_info':True,'side_info':{1:o.m_side.to(cuda_device)},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5,'deep':False } },
    #             # 2:{'ii':2,'lambda':0.0001,'r_1':10,'n_list':[o.data.shape[2]],'r_2':1,'has_side_info':True,'side_info':{1:o.t_side.to(cuda_device)},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5,'deep':False} }
    #              }
    # model = KFT(init_dict,lambda_reg=1e-5,cuda=cuda_device).to(cuda_device)
    # # model = variational_KFT(init_dict,KL_weight=1.,cuda=cuda_device).to(cuda_device)
    #
    # ITS = 200
    # opt = torch.optim.Adam(model.parameters(),lr=1e-2) #"some weird ass bug"
    # loss_func = torch.nn.MSELoss()
    # EPOCHS = 10
    # for j in range(EPOCHS):
    #     model.turn_off_kernel_mode()
    #     for param_group in opt.param_groups:
    #         param_group['lr'] = 1e-2
    #     # print_model_parameters(model)
    #     for i in range(ITS):
    #         X,Y = o.get_batch(1.0)
    #         if cuda_device is not None:
    #             Y=Y.to(cuda_device)
    #         y_pred, reg = model(X)
    #         risk_loss = loss_func(y_pred,Y)
    #         loss =  risk_loss+reg
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #         # with torch.no_grad():
    #         #     mean_preds = model.mean_forward(X)
    #         #     mean_risk_loss = loss_func(mean_preds,Y)
    #         # print(mean_risk_loss.data)
    #         print(risk_loss.data)
    #         print('-')
    #         print(reg.data)
    #     model.turn_on_kernel_mode()
    #     for param_group in opt.param_groups:
    #         param_group['lr'] = 1e-3
    #     # print_model_parameters(model)
    #
    #     for i in range(ITS):
    #         X, Y = o.get_batch(1.0)
    #         if cuda_device is not None:
    #             Y = Y.to(cuda_device)
    #         y_pred, reg = model(X)
    #         risk_loss = loss_func(y_pred, Y)
    #         loss = risk_loss + reg
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #         # with torch.no_grad():
    #         #     mean_preds = model.mean_forward(X)
    #         #     mean_risk_loss = loss_func(mean_preds,Y)
    #         # print(mean_risk_loss.data)
    #         print(risk_loss.data)
    #         print('-')
    #         print(reg.data)


