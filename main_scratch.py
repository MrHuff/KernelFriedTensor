import torch
import gpytorch
import pykeops
from tensorly.tenalg import mode_dot,multi_mode_dot
import tensorly
tensorly.set_backend('pytorch')
from pykeops.torch import LazyTensor as keops
from tensorly.base import fold
from KFT.util import kernel_adding_tensor
from KFT.KFT import KFT
def keops_mode_product(T,K,mode):
    """
    :param T: Pytorch tensor, just pass as usual
    :param K: Keops Lazytensor object, remember should be on form MxN
    :param mode: Mode of tensor
    :return:
    """
    t_new_shape = list(T.shape)
    t_new_shape[mode] = K.shape[0]
    T = K @ torch.reshape(torch.transpose(T, mode, 0), (T.shape[mode], -1))
    T = fold(unfolded_tensor=T,mode=mode,shape=t_new_shape)
    return T

# def lazy_ones(n):
#     x = keops(torch.ones(*(n,1)),axis=0)
#     y = keops(torch.ones(*(n,1)),axis=1)
#     ones = (x*y).sum(dim=2)
#     return ones

# def MSE_loss(pred,target):
#     return torch.sum(((1*pred-1*target)*1)**2)


if __name__ == '__main__':


    print(pykeops.bin_folder)
    PATH = './experiment_3/'
    #Doesn't like last index being 1 -> do squeeze type operation!
    o = kernel_adding_tensor(PATH)
    print(o.data.shape[0])
    print(o.data.shape[1])
    print(o.data.shape[2])
    init_dict = {
                 0:{'ii':0,'lambda':0.01,'r_1':1,'n_list':[o.data.shape[0]],'r_2':10,'has_side_info':True,'side_info':{1:o.n_side},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5} },
                 1:{'ii':1,'lambda':0.01,'r_1':10,'n_list':[o.data.shape[1]],'r_2':10,'has_side_info':True,'side_info':{1:o.m_side},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5} },
                 2:{'ii':2,'lambda':0.01,'r_1':10,'n_list':[o.data.shape[2]],'r_2':1,'has_side_info':True,'side_info':{1:o.t_side},'kernel_para':{'ls_factor':1.0, 'kernel_type':'rbf','nu':2.5} }
                 }

    cuda_device = 'cuda:0'
    model = KFT(init_dict,cuda=cuda_device).to(cuda_device)
    # for n,p in model.named_parameters():
    #     print(n,)
    #     print(p)
    ITS = 1
    opt = torch.optim.Adam(model.parameters(),lr=0.01)
    loss_func = torch.nn.MSELoss()

    for i in range(ITS):
        X,Y = o.get_batch(0.001)
        if cuda_device is not None:
            Y=Y.to(cuda_device)
        y_pred, reg = model(X)
        print(reg)
        loss = reg#loss_func(y_pred,Y) #+ reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.data)
###Cant be combined with gpytorch yet, must write custom kernels...
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