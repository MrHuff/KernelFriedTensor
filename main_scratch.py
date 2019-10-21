import torch
import gpytorch
from tensorly.tenalg import mode_dot,multi_mode_dot
import tensorly
tensorly.set_backend('pytorch')
from pykeops.torch import LazyTensor as keops
from tensorly.base import fold

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



###Cant be combined with gpytorch yet, must write custom kernels...
# kernel = gpytorch.kernels.keops.MaternKernel()
# kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor([1.]),requires_grad=False)
#
# t = torch.nn.Parameter(torch.randn(*(2,10000,2)),requires_grad=False)
# print(t.index_select(dim=1,index=torch.tensor([0,1,2,3])).shape)
#
# A = torch.randn(*(1000000,5))
# B = A #A[0,:].unsqueeze(0)
# # x = keops(B,axis=0)
# # y = keops(A,axis=1)
# eval_kernel =   kernel(B,A) #keops_rbf(x,y,ls=1.)
#
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



n_list = [3,3]
permutation_list = [i+1 for i in range(len(n_list))] + [0,-1]
print(permutation_list)
t = torch.randn(*(2,3,3,2))
print(t.shape)
indices = torch.tensor([[0,0],[0,1],[0,1],[1,0],[0,0]])
# indices = torch.tensor([0,1,0,0])
_t = t.permute(permutation_list)[indices.unbind(1)]#.transpose(0,1)
print(_t.shape)
print(_t)
# sanity = t[:,indices[:,0],indices[:,1],:].transpose(0,1)
# print(sanity)