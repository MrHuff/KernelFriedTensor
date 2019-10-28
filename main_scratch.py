import torch
import gpytorch
import pykeops
from tensorly.tenalg import mode_dot,multi_mode_dot
import tensorly
tensorly.set_backend('pytorch')
from pykeops.torch import LazyTensor as keops
from tensorly.base import fold
from KFT.util import kernel_adding_tensor
from KFT.KFT_keops import keops_RBF
from pykeops.torch import KernelSolve
from KFT.KFT_fp_16 import KFT,variational_KFT
import time
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
    # gpytorch.utils.StochasticLQ()
    n = 5000
    a = torch.randn(*(n,10)).cuda()
    C = a@a.t() + torch.eye(n).cuda()
    start = time.time()
    print((a*a).sum()+n)
    end = time.time()
    print(end-start)

    # start = time.time()
    # L = C.cholesky()
    # print(L.diag().prod()**2)
    # end = time.time()
    # print(end - start)

    start = time.time()
    print(C.det())
    end = time.time()
    print(end-start)

    start = time.time()
    print(gpytorch.log_det(C))
    end = time.time()
    print(end-start)



#     #TODO: do VI! Try FP16 implementation. Introduce numerical reguarlization.
#     # test_k = gpytorch.kernels.keops.RBFKernel().cuda()
#     device  = 'cuda:0'
#     # test_k = keops_RBF()
#     # test_k.raw_lengthscale = torch.nn.Parameter(torch.tensor(1.).cuda(),requires_grad=False)
#
#     print(pykeops.bin_folder)
#     PATH = './experiment_3/'
#     #Doesn't like last index being 1 -> do squeeze type operation!
#     o = kernel_adding_tensor(PATH)
#     print(o.data.shape[0])
#     print(o.data.shape[1])
#     print(o.data.shape[2])
#
#     # N=10000
#     # D=5
#     # Dv=2
#     # X = torch.randn(D,1).cuda()
#     # x = torch.rand(N, D, device=device)
#     # b = torch.randn(N, Dv, device=device)
#     # gamma = torch.ones(1, device=device) * .5 / .01 ** 2  # kernel bandwidth
#     # alpha = torch.ones(1, device=device) * 0.8  # regularization
#     # res = Kinv_keops(x, b, gamma, alpha)
#     # print(res)
#     cuda_device = 'cuda:0'
#
#     init_dict = {
#                 # 0: {'ii': [0, 1], 'lambda': 1e-3, 'r_1': 1, 'n_list': [o.data.shape[0], o.data.shape[1]], 'r_2': 10,'has_side_info': True, 'side_info': {1:o.n_side,2:o.m_side},'kernel_para': {'ls_factor': 1.0, 'kernel_type': 'rbf', 'nu': 2.5}},
#
#                 0:{'ii':0,'lambda':1e-6,'r_1':1,'n_list':[o.data.shape[0]],'r_2':10,'has_side_info':True,'side_info':{1:o.n_side},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5,'deep':False} },
#                 # 1:{'ii':[2],'lambda':1e-3,'r_1':10,'n_list':[o.data.shape[2]],'r_2':1,'has_side_info':True,'side_info':{1:o.t_side},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5} },
#                 # 1:{'ii':[1,2],'lambda':1e-6,'r_1':10,'n_list':[o.data.shape[1],o.data.shape[2]],'r_2':1,'has_side_info':True,'side_info':{1:o.m_side,2:o.t_side},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5} },
#                 1:{'ii':1,'lambda':0.0001,'r_1':10,'n_list':[o.data.shape[1]],'r_2':10,'has_side_info':True,'side_info':{1:o.m_side},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5,'deep':False } },
#                 2:{'ii':2,'lambda':0.0001,'r_1':10,'n_list':[o.data.shape[2]],'r_2':1,'has_side_info':True,'side_info':{1:o.t_side},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5,'deep':False} }
#                  }
#     #
#     model = KFT(init_dict,cuda=cuda_device).to(cuda_device)
#     # model = variational_KFT(init_dict,KL_weight=1.,cuda=cuda_device).to(cuda_device)
#
#     # for n,p in model.named_parameters():
#     #     print(n,)
#     #     print(p.shape)
#     #     print(p.device)
#     ITS = 1000
#     opt = torch.optim.Adam(model.parameters(),lr=1e-2) #"some weird ass bug"
#     loss_func = torch.nn.MSELoss()
#
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
# #
# ###Cant be combined with gpytorch yet, must write custom kernels...
# kernel = gpytorch.kernels.keops.MaternKernel()
# kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor([1.]),requires_grad=False)
# n = 100
# t = torch.nn.Parameter(torch.randn(*(2,n,2)),requires_grad=False)
# ones = lazy_ones(n)
# test = keops_mode_product(t*t,ones,mode=1)
# print(test)
# print(test.shape)

# A = torch.randn(*(n,5))
# B = A #A[0,:].unsqueeze(0)
# ls = torch.nn.Parameter(torch.tensor(1.0),requires_grad=False)
# p = torch.nn.Parameter(torch.tensor(1.0),requires_grad=False)
# p_kernel = keops_periodic(ls=ls,p=p)
# eval_kernel = p_kernel(A,B)
# print(eval_kernel)
# # x = keops(B,axis=0)
# # y = keops(A,axis=1)
# eval_kernel =   kernel(B,A) #keops_rbf(x,y,ls=1.)

# test = keops_mode_product(t,eval_kernel,mode=1)
# print(test.shape)
# print(test)


# kernel_tensorly = kernel(B,A).evaluate()
# tensorly_ref = mode_dot(t,kernel_tensorly,mode = 1)
# print(tensorly_ref)




# print(t_lazy.shape)
# # eval_kernel = kernel(A[0:5,:],A).unsqueeze(0)
# # k_shape = eval_kernel.shape[1:]
# print(eval_kernel.shape)
# test = t_lazy.keops_tensordot(eval_kernel,t_shape,eval_kernel.shape,(1),(0),(2,1,5,2))
# print(test)
# a = torch.arange(60.).reshape(3, 4, 5)
# b = torch.arange(24.).reshape(4, 3, 2)
# c = torch.tensordot(a, b, dims=([1, 0], [0, 1]))
# print(c.shape)
#
# t = torch.randn(*(2,10,2))
# # t_lazy = LazyTensor(t[:,:,None,:])
# A = torch.randn(*(10,5))
# kernel = gpytorch.kernels.keops.RBFKernel()
# eval_kernel = kernel(A[0:5,:],A)
# print(eval_kernel.shape)
# print(t.shape)
# y = torch.tensordot(eval_kernel.evaluate(),t,dims =([1],[1]))
# y = y.permute(1,0,2)
# print(y)
# y_tensorly = mode_dot(t,eval_kernel.evaluate(),mode=1)
# print(y_tensorly)



# n_list = [3,3]
# permutation_list = [i+1 for i in range(len(n_list))] + [0,-1]
# print(permutation_list)
# t = torch.randn(*(2,3,3,2))
# print(t.shape)
# indices = torch.tensor([[0,0],[0,1],[0,1],[1,0],[0,0]])
# # indices = torch.tensor([0,1,0,0])
# _t = t.permute(permutation_list)[indices.unbind(1)]#.transpose(0,1)
# print(_t.shape)
# print(_t)
# sanity = t[:,indices[:,0],indices[:,1],:].transpose(0,1)
# print(sanity)