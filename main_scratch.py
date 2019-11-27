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

from KFT.benchmarks.lgbm import lgbm

if __name__ == '__main__':
    params= dict()
    params['its'] = 10
    params['hyperopts'] = 20
    params['regression'] = True
    SAVE_PATH = './lgbm_test'
    seed = 1
    y_name = 'total_sales'
    data_path = 'benchmark_data_lgbm'
    l = lgbm(seed=seed,y_name=y_name,data_path=data_path,save_path=SAVE_PATH,params=params)
    l.run()



