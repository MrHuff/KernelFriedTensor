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

import torch
from sklearn.preprocessing import StandardScaler
from KFT.util import process_old_setup,concat_old_side_info
from KFT.util import post_process

if __name__ == '__main__':
    # post_process('./private_job_arch_0/','test_R2')
    # n,m,t = torch.load('./tensor_data/side_info.pt')
    # print(t)
    # for name in ['location','article','time']:
    #     location = torch.load(f'./tensor_data/{name}_tensor_400000.pt').numpy()
    #     scaler_location = StandardScaler()
    #     location_scaled = torch.tensor(scaler_location.fit_transform(location))
    #     torch.save(location_scaled, f'./tensor_data/{name}_tensor_400000_scaled.pt')
    # process_old_setup('./tensor_data/','data_tensor_400000.pt')
    concat_old_side_info('./tensor_data/',['location_tensor_400000_scaled.pt','article_tensor_400000_scaled.pt','time_tensor_400000.pt'])
