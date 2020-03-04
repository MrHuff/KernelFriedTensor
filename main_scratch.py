





# from pykeops.torch import KernelSolve
# from KFT.job_utils import run_job_func
# import warnings
# from KFT.KFT_fp_16 import edge_mode_product,row_outer_prod
# import torch
# from tensorly.tenalg import multi_mode_dot,mode_dot
# import numpy as np
# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)
#
# def Kinv_keops(x, b, gamma, alpha):
#     N=10000
#     D=5
#     Dv=2
#     formula = 'Exp(- g * SqDist(x,y)) * a'
#     aliases = ['x = Vi(' + str(D) + ')',  # First arg:  i-variable of size D
#                'y = Vj(' + str(D) + ')',  # Second arg: j-variable of size D
#                'a = Vj(' + str(Dv) + ')',  # Third arg:  j-variable of size Dv
#                'g = Pm(1)']  # Fourth arg: scalar parameter
#     Kinv = KernelSolve(formula, aliases, "a", axis=1)
#     res = Kinv(x, x, b, gamma, alpha=alpha)
#     return res
#
# if __name__ == '__main__':
#     # X = torch.randn(*(5,5,5))
#     # b = torch.randn(*(10,5))
#     # d = b.t().unsqueeze(-1)
#     # print(d.shape)
#     # res = mode_dot(X,b,mode=2)
#     # print(res)
#     # print(res.shape)
#     # res_2 = edge_mode_product(X,d,2,0).squeeze()
#     # print(res_2)
#     # print(res_2.shape)
#     # print(res_2==res)
#     a = torch.tensor(np.arange(1,11)).expand(10,-1)
#     c = row_outer_prod(a,2*a)
#     c_2 = row_outer_prod(2*a,a)
#     print((c_2==c).all())
#     # a = torch.tensor(1e-6).half().cuda()
#     # b = torch.tensor(1e-6).half().cuda()
#     # print(a+b)

# class keops_RBFkernel(torch.nn.Module):
#     def __init__(self,ls,x,y=None,):
#         super(keops_RBFkernel, self).__init__()
#         self.raw_lengthscale = torch.nn.Parameter(ls,requires_grad=False).contiguous()
#         self.raw_lengthscale.requires_grad = False
#         self.register_buffer('x', x.contiguous())
#         self.shape = (x.shape[0],x.shape[0])
#         if y is not None:
#             self.register_buffer('y',y.contiguous())
#         else:
#             self.y = None
#
#     def __call__(self, *args, **kwargs):
#         return self
#
#     def __matmul__(self, b):
#         s = self.forward(b)
#         return s
#
#     def forward(self,b):
#         params = {
#         "id": Kernel("gaussian(x,y)"),
#         "gamma": .5 / self.raw_lengthscale ** 2,
#         }
#
#         if self.y is not None:
#             return kernel_product(params, self.x, self.y, b,mode='sum',backend='pytorch')
#         else:
#             return kernel_product(params, self.x, self.x, b,mode='sum',backend='pytorch')
from pykeops.torch import LazyTensor,Genred

def get_formula(nu,D,Dv):
    aliases = ['G_0 = Pm(0, ' + str(D) + ')',
               'X_0 = Vi(1, ' + str(D) + ')',  # First arg:  i-variable of size D
               'Y_0 = Vj(2, ' + str(D) + ')',  # Second arg: j-variable of size D
               'B_0 = Vj(3, ' + str(Dv) + ')',  # Third arg:  j-variable of size Dv
               ]  # Fourth arg: scalar parameter
    if nu==0.5:
        # (Exp(-Sqrt( WeightedSqDist(G_0,X_0,Y_0))) * B_0) #CORRECT
        formula = '(Exp(-Sqrt( WeightedSqDist(G_0,X_0,Y_0))) * B_0)'
    elif nu==1.5:
        aliases.append('g = Pm(1)')
        formula = '((IntCst(1)+g*Sqrt( WeightedSqDist(G_0,X_0,Y_0)))*Exp(-g*Sqrt( WeightedSqDist(G_0,X_0,Y_0))) * B_0)'
    elif nu==2.5:
        aliases.append('g = Pm(1)')
        aliases.append('g_2 = Pm(1)')
        formula = '((IntCst(1)+g*Sqrt( WeightedSqDist(G_0,X_0,Y_0))+g_2*WeightedSqDist(G_0,X_0,Y_0))*Exp(-g*Sqrt( WeightedSqDist(G_0,X_0,Y_0))) * B_0)'
    gen_formula = Genred(formula, aliases,  reduction_op='Sum', axis=1,dtype='float32')
    return gen_formula

def run_formula(formula,nu,x,y,g,b,device_id=0):
    if nu == 0.5:
        return formula(*[g,x,y,b],backend='GPU',device_id=device_id)
    elif nu == 1.5:
        c_1 = torch.tensor([3.0]).sqrt()
        return formula(*[g,x,y,b,c_1],backend='GPU',device_id=device_id)
    elif nu == 2.5:
        c_1 = torch.tensor([5.]).sqrt()
        c_2 = torch.tensor([5./3.])
        return formula(*[g,x,y,b,c_1,c_2],backend='GPU',device_id=device_id)


if __name__ == '__main__':
    import torch
    import time
    from pykeops.torch.kernel_product.kernels import Kernel, kernel_product
    import gpytorch
    import pykeops
    # pykeops.clean_pykeops()  # just in case old build files are still present
    x = torch.randn(100000, 10, requires_grad=True) #Keops really slow for "wide matrices". Needs to be adjusted.
    # y = torch.randn(2000, 3, requires_grad=True)
    b = torch.randn(100000, 100, requires_grad=True)

    # print(test.shape)
    #
    # print(b.shape)




    # test_k = gpytorch.kernels.keops.MaternKernel(ard_num_dims=10,nu=2.5)
    # s = time.time()
    # beta = test_k(x,x)@b
    # e = time.time()
    # print(e-s)
    # formula: Sum_Reduction((Exp(-(WeightedSqDist(G_0, X_0, Y_0))) * B_0), 0)
    # aliases: G_0 = Pm(0, 10);
    # X_0 = Vi(1, 10);
    # Y_0 = Vj(2, 10);
    # B_0 = Vj(3, 100);
    # dtype: float32

    # sigma = torch.nn.Parameter(torch.tensor([.5]*10),requires_grad=True)
    # print(sigma.shape)
    # params = {
    #     "id": Kernel("laplacian(x,y)"),
    #     "gamma": .5 / sigma ** 2,
    # }
    # s = time.time()
    # a = kernel_product(params, x, x, b)
    # e = time.time()
    # print(e-s)

    g = torch.tensor([0.5]*10)
    nu = 2.5
    formula = get_formula(nu=nu, D=10, Dv=100)
    s = time.time()
    test = run_formula(formula=formula, x=x, y=x, g=g, b=b, nu=nu, device_id=0)
    e = time.time()
    print(e-s)
    print(test.shape)

    # print(a)
    # torch.dot(a.view(-1), torch.ones_like(a).view(-1)).backward()
    # print(sigma.grad)
    # post_process('./private_job_arch_0/','test_R2')
    # n,m,t = torch.load('./tensor_data/side_info.pt')
    # print(t)
    # for name in ['location','article','time']:
    #     location = torch.load(f'./tensor_data/{name}_tensor_400000.pt').numpy()
    #     scaler_location = StandardScaler()
    #     location_scaled = torch.tensor(scaler_location.fit_transform(location))
    #     torch.save(location_scaled, f'./tensor_data/{name}_tensor_400000_scaled.pt')
    # process_old_setup('./tensor_data/','data_tensor_400000.pt')
    # concat_old_side_info('./tensor_data/',['location_tensor_400000_scaled.pt','article_tensor_400000_scaled.pt','time_tensor_400000.pt'])
