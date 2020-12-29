import torch
import tensorly
tensorly.set_backend('pytorch')
import gpytorch
from tensorly.base import fold,unfold,partial_fold
import math
import time
from torch.distributions import StudentT
from pykeops.torch import LazyTensor,Genred
PI  = math.pi
torch.set_printoptions(profile="full")

class keops_RBFkernel(torch.nn.Module):
    def __init__(self,ls,x,y=None,device_id=0):
        super(keops_RBFkernel, self).__init__()
        self.device_id = device_id
        self.raw_lengthscale = torch.nn.Parameter(ls,requires_grad=False).contiguous()
        self.raw_lengthscale.requires_grad = False
        self.register_buffer('x', x.contiguous())
        self.shape = (x.shape[0],x.shape[0])
        if y is not None:
            self.register_buffer('y',y.contiguous())
        else:
            self.y = x
        self.gen_formula = None

    def get_formula(self,D,ls_size,Dv):
        aliases = ['G_0 = Pm(0, ' + str(ls_size) + ')',
                   'X_0 = Vi(1, ' + str(D) + ')',
                   'Y_0 = Vj(2, ' + str(D) + ')',
                   'B_0 = Vj(3, ' + str(Dv) + ')'
                   ]
        formula = '(Exp( -WeightedSqDist(G_0,X_0,Y_0)) * B_0)'
        return formula,aliases

    def evaluate(self):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __matmul__(self, b):
        s = self.forward(b)
        return s

    def forward(self,b):
        Dv = b.shape[1]
        if self.gen_formula is None:
            self.formula, self.aliases = self.get_formula(D=self.x.shape[1], ls_size=self.raw_lengthscale.shape[0],Dv=Dv)
            self.gen_formula = Genred(self.formula, self.aliases, reduction_op='Sum', axis=1, dtype='float32')
        return self.gen_formula(*[self.raw_lengthscale,self.x,self.y,b],backend='GPU',device_id=self.device_id)

class keops_matern_kernel(keops_RBFkernel):

    def __init__(self,ls,x,y=None,nu=0.5,device_id = 0):
        super(keops_matern_kernel, self).__init__(ls,x,y,device_id)
        self.nu = nu
        if self.nu == 1.5:
            self.register_buffer('c_1',torch.tensor([3.0]).sqrt())
        elif self.nu==2.5:
            self.register_buffer('c_1',torch.tensor([5.0]).sqrt())
            self.register_buffer('c_2',torch.tensor([5.0/3.0]).sqrt())

    def get_formula_matern(self,nu, D,ls_size,Dv):
        aliases = ['G_0 = Pm(0, ' + str(ls_size) + ')',
                   'X_0 = Vi(1, ' + str(D) + ')',  # First arg:  i-variable of size D
                   'Y_0 = Vj(2, ' + str(D) + ')',  # Second arg: j-variable of size D
                   'B_0 = Vj(3, ' + str(Dv) + ')',
                   ]  # Fourth arg: scalar parameter
        if nu == 0.5:
            formula = '(Exp(-Sqrt( WeightedSqDist(G_0,X_0,Y_0))) * B_0)'
        elif nu == 1.5:
            aliases.append('g = Pm(4,1)')
            formula = '((IntCst(1)+g*Sqrt( WeightedSqDist(G_0,X_0,Y_0)))*Exp(-g*Sqrt( WeightedSqDist(G_0,X_0,Y_0))) * B_0)'
        elif nu == 2.5:
            aliases.append('g = Pm(4,1)')
            aliases.append('g_2 = Pm(5,1)')
            formula = '((IntCst(1)+g*Sqrt( WeightedSqDist(G_0,X_0,Y_0))+g_2*WeightedSqDist(G_0,X_0,Y_0))*Exp(-g*Sqrt( WeightedSqDist(G_0,X_0,Y_0))) * B_0)'

        return formula,aliases

    def forward(self,b):
        Dv = b.shape[1]
        if self.gen_formula is None:
            self.formula, self.aliases = self.get_formula_matern(nu=self.nu, D=self.x.shape[1],
                                                                 ls_size=self.raw_lengthscale.shape[0],Dv=Dv)
            self.gen_formula = Genred(self.formula, self.aliases, reduction_op='Sum', axis=1, dtype='float32')
        if self.nu==0.5:
            return self.gen_formula(*[self.raw_lengthscale,self.x,self.y,b],backend='GPU',device_id=self.device_id)
        elif self.nu==1.5:
            return self.gen_formula(*[self.raw_lengthscale,self.x,self.y,b,self.c_1],backend='GPU',device_id=self.device_id)
        elif self.nu==2.5:
            return self.gen_formula(*[self.raw_lengthscale,self.x,self.y,b,self.c_1,self.c_2],backend='GPU',device_id=self.device_id)


def transpose_khatri_rao(x, y):
    """
    :param x: n x c
    :param y: n x d
    :return: n x (cd)
    """
    y = y.unsqueeze(-1).expand(-1,-1,x.shape[1]).permute(0,2,1)
    return torch.einsum('ij,ijk->ijk', x, y).flatten(1)

def edge_mode_product(T,also_T,mode_T, mode_also_T):
    """
    :param T: Tensor - core [n_1,n_2.n_3]
    :param also_T: Tensor - to be applied [n_3,m_2,m_3]
    :param mode: to which mode of T
    :return: mode = 2 -> [n_1,n_2,m_2,m_3]
    """
    final_shape = list(T.shape[0:(mode_T)]) + list(also_T.shape[mode_also_T+1:])
    also_T = unfold(also_T,mode_also_T)
    T = lazy_mode_product(T,also_T.t(),mode_T)
    return T.view(final_shape)

def lazy_mode_product(T, K, mode):
    """
    :param T: Pytorch tensor, just pass as usual
    :param K: Gpytorch lazy tensor, remember should be on form MxN
    :param mode: Mode of tensor
    :return:
    """
    new_shape = list(T.shape)
    new_shape[mode] = K.shape[0]
    T = unfold(T,mode)
    T = K@T.contiguous()
    T = fold(T,mode,new_shape)
    return T

def lazy_mode_hadamard(T,vec,mode):
    """
    :param T: Pytorch tensor, just pass as usual
    :param vec: A vector to be applied hadamard style, is equivalent of diagonalizing vec and doing matrix product.
    :param mode: Mode of tensor
    :return:
    """
    new_shape = list(T.shape)
    T = unfold(T,mode)
    T = vec*T
    T = fold(T,mode,new_shape)
    return T

class RFF(torch.nn.Module):
    def __init__(self, X, lengthscale, rand_seed=1,kernel='rbf',kernel_para=None):
        super(RFF, self).__init__()
        torch.random.manual_seed(rand_seed)
        self.n_input_feat = X.shape[1] # dimension of the original input
        self.n_feat = int(round((X.shape[0]**0.25)*math.log(X.shape[0])))#Michaels paper!
        print(f'I= {self.n_feat}')
        self.raw_lengthscale = torch.nn.Parameter(lengthscale, requires_grad=False)
        if kernel=='rbf':
            self.register_buffer('w' ,torch.randn(*(self.n_feat,self.n_input_feat)))
        elif kernel=='matern':
            dist = StudentT(df=kernel_para['nu'])
            self.register_buffer('w' ,dist.sample((self.n_feat,self.n_input_feat)))

        self.register_buffer('b' ,torch.rand(*(self.n_feat, 1))*2.0*PI)
        self.register_buffer('X',X)

    def __call__(self, *args, **kwargs):
        return self

    def evaluate(self):
        return torch.transpose(
            math.sqrt(2. / float(self.n_feat)) * torch.cos(torch.mm(self.w / self.raw_lengthscale, self.X.t()) + self.b), 0,
            1)

class KFTR_temporal_regulizer(torch.nn.Module):
    def __init__(self,r_1,n_list,r_2,time_idx,base_ref_int,lag_set_tensor,lambda_W):
        super(KFTR_temporal_regulizer, self).__init__()
        self.T = n_list[time_idx]
        self.time_idx = time_idx
        n_list[time_idx] = lag_set_tensor.shape[0]
        W_size = [r_1,*n_list,r_2]
        self.W = torch.nn.Parameter( torch.randn(*W_size).squeeze().float(),requires_grad=True)
        self.base_ref = base_ref_int
        self.lambda_W = lambda_W
        self.lag_tensor = lag_set_tensor
        self.loss = torch.nn.MSELoss()

    def freeze_param(self):
        self.W.requires_grad=False

    def activate_param(self):
        self.W.requires_grad=True

    def forward(self,index,time_component):
        offset = index-self.base_ref
        if offset<0:
            return time_component.index_select(self.time_idx,index).squeeze()
        else:
            lags  = self.lag_tensor + offset
            return self.W*time_component.index_select(self.time_idx,lags).sum(dim=self.time_idx)

    def calculate_square_error(self,actual_component,predicted_component):
        return self.loss(actual_component,predicted_component)

    def calculate_KFTR(self,time_component):
        KFTR = 0
        for idx in range(self.base_ref,self.T):
            x_t = time_component.index_select(self.time_idx,idx)
            x_t_pred = self.forward(idx,time_component)
            KFTR+=self.calculate_square_error(actual_component=x_t,predicted_component=x_t_pred)
        return KFTR

    def get_reg(self):
        return self.lambda_W*torch.mean(self.W**2)


class TT_component(torch.nn.Module):
    def __init__(self, r_1, n_list, r_2, cuda=None, config=None, init_scale=1.0, old_setup=False, reg_para=0, double_factor=False, sub_R=2):
        super(TT_component, self).__init__()
        self.bayesian = config['bayesian']
        self.register_buffer('reg_para',torch.tensor(reg_para))
        self.double_factor = double_factor
        self.r_1 = r_1
        self.r_2 = r_2
        self.n_list = n_list
        self.amp = None
        self.device = cuda
        self.old_setup = old_setup
        self.full_grad = config['full_grad']
        self.dual = config['dual']
        self.n_dict = {i + 1: None for i in range(len(n_list))}
        self.RFF_dict = {i + 1: False for i in range(len(n_list))}
        self.shape_list  = [r_1]+[n for n in n_list] + [r_2]
        self.permutation_list = [i + 1 for i in range(len(n_list))] + [0, -1]
        if self.double_factor:
            self.core_param = sub_factorization(self.shape_list,R=sub_R,init_scale=init_scale)
        else:
            self.core_param = torch.nn.Parameter(init_scale*torch.ones(*self.shape_list), requires_grad=True)

        self.init_scale = init_scale
        for i, n in enumerate(n_list):
            self.register_buffer(f'reg_ones_{i}',torch.ones((n,1)))

    def index_select(self,indices,T):
        return T.permute(self.permutation_list)[indices]

    def turn_off(self):
        for p in self.parameters():
            p.requires_grad = False

    def turn_on(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self,indices):
        if self.double_factor:
            p = self.core_param()
        else:
            p = self.core_param
        if self.dual and not self.old_setup:
            reg = self.get_aux_reg_term()
        else:
            reg = p**2
        if self.full_grad:
            return p, reg
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return p.permute(self.permutation_list)[indices], reg

    def forward_scale(self,indices):
        if self.full_grad:
            return self.core_param, self.core_param**2
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return self.core_param.permute(self.permutation_list)[indices], self.core_param**2

    def get_aux_reg_term(self):
        if self.old_setup:
            return 1.
        else:
            if self.double_factor:
                p = self.core_param()
            else:
                p = self.core_param
            T = p ** 2
            for mode,ones in enumerate(self.n_list):
                ones = getattr(self,f'reg_ones_{mode}')
                T = lazy_mode_product(T, ones.t(), mode+1)
                T = lazy_mode_product(T, ones, mode+1)
            return T

    def toggle_mean_var(self,toggle):
        self.core_param.requires_grad = toggle
        self.variance_parameters.requires_grad = not toggle

class TT_kernel_component(TT_component): #for tensors with full or "mixed" side info
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_para_dict,cuda=None,config=None,init_scale=1.0,reg_para=0,old_setup=False):
        super(TT_kernel_component, self).__init__(r_1,n_list,r_2,cuda,config,init_scale,old_setup,reg_para)
        self.core_param = torch.nn.Parameter(init_scale*torch.randn(*self.shape_list), requires_grad=True)
        self.kernel_eval_mode = False
        self.side_info_dict = {}
        if self.device not in ['cpu']:
            for key, val in side_information_dict.items():
                self.side_info_dict[key] = val.to(self.device)
        for key,_ in self.side_info_dict.items(): #Should be on the form {mode: side_info}'
            if self.dual:
                self.assign_kernel(key,kernel_para_dict)
            else:
                self.set_side_info(key)

    def set_side_info(self,key):
        with torch.no_grad():
            if self.dual:
                tmp_kernel_func = getattr(self, f'kernel_{key}')
                self.n_dict[key] = tmp_kernel_func(self.side_info_dict[key]).evaluate()
            else:
                self.n_dict[key] = self.side_info_dict[key]

    def kernel_train_mode_on(self):
        self.turn_off()
        self.kernel_eval_mode = True
        if self.dual:
            for key,val in self.n_dict.items():
                if val is not None:
                    k = getattr(self,f'kernel_{key}')
                    k.raw_lengthscale.requires_grad = True
                    if k.__class__.__name__=='PeriodicKernel':
                        k.raw_period_length.requires_grad=True
            return 0

    def kernel_train_mode_off(self):
        self.turn_on()
        self.kernel_eval_mode = False
        if self.dual:
            for key,val in self.n_dict.items():
                if val is not None:
                    k = getattr(self,f'kernel_{key}')
                    k.raw_lengthscale.requires_grad = False
                    if k.__class__.__name__=='PeriodicKernel':
                        k.raw_period_length.requires_grad=False
                    self.set_side_info(key)
        return 0

    def get_median_ls(self,key):  # Super LS and init value sensitive wtf
        with torch.no_grad():
            base = gpytorch.kernels.Kernel()
            X = self.side_info_dict[key]
            if X.shape[0] > 5000:
                idx = torch.randperm(5000)
                X = X[idx, :]
            d = base.covar_dist(X, X)
            return torch.sqrt(torch.median(d[d > 0])).unsqueeze(0).cpu()

    def assign_kernel(self,key,kernel_dict_input):
        kernel_para_dict = kernel_dict_input[key]
        self.gamma_sq_init = self.get_median_ls(key)
        ard_dims = None if not kernel_para_dict['ARD'] else self.side_info_dict[key].shape[1]
        if self.side_info_dict[key].shape[1] > 15 and self.side_info_dict[key].shape[0] > 10000:
            self.RFF_dict[key] = True
        if self.RFF_dict[key]:
            setattr(self, f'kernel_{key}', RFF(self.side_info_dict[key],
                                               lengthscale=self.gamma_sq_init,
                                               kernel=kernel_para_dict['kernel_type'],
                                               kernel_para=kernel_para_dict).to(self.device))
        else:
            if kernel_para_dict['kernel_type']=='rbf':
                setattr(self, f'kernel_{key}', gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims).to(self.device))
            elif kernel_para_dict['kernel_type']=='matern':
                setattr(self, f'kernel_{key}', gpytorch.kernels.MaternKernel(ard_num_dims=ard_dims,nu=kernel_para_dict['nu']).to(self.device))

            elif kernel_para_dict['kernel_type']=='periodic':
                setattr(self, f'kernel_{key}', gpytorch.kernels.PeriodicKernel(ard_num_dims=ard_dims).to(self.device))
                getattr(self, f'kernel_{key}').raw_period_length=torch.nn.Parameter(torch.tensor(kernel_para_dict['p']).to(self.device),
                                                                                requires_grad=False)


            elif kernel_para_dict['kernel_type'] == 'local_periodic':
                setattr(self, f'kernel_{key}', gpytorch.kernels.PeriodicKernel(ard_num_dims=ard_dims).to(self.device))
                getattr(self, f'kernel_{key}').raw_period_length=torch.nn.Parameter(torch.tensor(kernel_para_dict['p']).to(self.device),
                                                                                requires_grad=False)

            ls_init = self.gamma_sq_init * torch.ones(*(1, 1 if ard_dims is None else ard_dims))
            getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(ls_init.to(self.device),
                                                                                requires_grad=False)
        self.set_side_info(key)

    def side_data_eval(self,key):
        tmp_kernel_func = getattr(self, f'kernel_{key}')
        if tmp_kernel_func.__class__.__name__ in ['RFF','keops_RBFkernel','keops_matern_kernel']:
            return tmp_kernel_func.evaluate()
        else:
            X =  self.side_info_dict[key].to(self.device)
            val = tmp_kernel_func(X).evaluate()
            return val

    def apply_kernels(self,T):
        for key, val in self.n_dict.items():
            if val is not None:
                if self.dual:
                    if self.kernel_eval_mode:
                        val = self.side_data_eval(key)
                    if not self.RFF_dict[key]:
                        T = lazy_mode_product(T, val, key)
                    else:
                        T = lazy_mode_product(T, val.t(), key)
                        T = lazy_mode_product(T, val, key)
                else:
                    T = lazy_mode_product(T, val, key)
        return T

    def forward(self,indices):
        """Do tensor ops"""
        T = self.apply_kernels(self.core_param)
        if self.dual:
            reg = T*self.core_param
        else:
            reg = self.core_param**2
        if self.full_grad:
            return T,reg
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices], reg  #return both to calculate regularization when doing frequentist

class sub_factorization(torch.nn.Module):
    def __init__(self,tensor_shape,R=2,init_scale=1):
        super(sub_factorization, self).__init__()
        self.tensor_shape = tensor_shape
        self.factor  = R**(len(tensor_shape)-1)
        self.init_scale = init_scale
        for i,n in enumerate(tensor_shape):
            if i==0:
                setattr(self,f'latent_component_{i}',torch.nn.Parameter(torch.ones(*(1,n,R) ),requires_grad=True))
            elif i==len(tensor_shape)-1:
                setattr(self,f'latent_component_{i}',torch.nn.Parameter(torch.ones(*(R,n,1) ),requires_grad=True))
            else:
                setattr(self,f'latent_component_{i}',torch.nn.Parameter(torch.ones(*(R,n,R) ),requires_grad=True))
    def forward(self):
        preds = getattr(self,f'latent_component_0')
        for i in range(1,len(self.tensor_shape)):
            m = getattr(self,f'latent_component_{i}')
            preds = edge_mode_product(preds, m, len(preds.shape) - 1, 0)  # General mode product!
        # print(preds.squeeze()/self.factor)
        return self.init_scale *preds.squeeze(0).squeeze(-1)/self.factor