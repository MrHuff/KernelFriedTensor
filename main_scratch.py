from pykeops.torch import KernelSolve
from KFT.job_utils import run_job_func
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
    job_path = './tensor_data/'
    save_path = './private_jobs_KFT/'
    params = {'PATH': job_path,
              'reg_para_a': 1e0,
              'reg_para_b': 1e0,
              'batch_size_a': 1.,
              'batch_size_b': 1.,
              'fp_16': True,
              'fused': True,
              'hyperits': 20,
              'save_path': save_path,
              'task': 'reg',
              'epochs': 10,
              'bayesian': False,
              'cuda': True,
              'full_grad': True,
              'sub_epoch_V': 100,
              'sub_epoch_ls': 100,
              'sub_epoch_prime': 100,
              'seed': 1,
              'side_info_order': [0, 1, 2],
              'temporal_tag': [2],
              'architecture': 0,
              }
    run_job_func(params)

