import torch
from KFT.KFT_fp_16 import KFT, variational_KFT,KFT_scale,varitional_KFT_scale
from torch.nn.modules.loss import _Loss
import gc
import pickle
import os
import warnings
from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK,STATUS_FAIL
from KFT.util import get_dataloader_tensor,print_model_parameters,get_free_gpu,process_old_setup,concat_old_side_info,load_side_info,print_ls_gradients
from sklearn import metrics
import time
import numpy as np
import multiprocessing as mp
import pandas as pd
import pykeops

cal_list = [5, 15, 25, 35, 45]
cal_string_list = [f'{cal}%' for cal in cal_list] + [f'{100-cal}%'for cal in reversed(cal_list)]
def get_non_lin(non_lin_name):
    if non_lin_name=='relu':
        return torch.nn.ReLU()
    elif non_lin_name=='tanh':
        return torch.nn.Tanh()
    elif non_lin_name=='leaky':
        return torch.nn.LeakyReLU()
    elif non_lin_name=='sig':
        return torch.nn.Sigmoid()
    elif non_lin_name=='linear':
        return lambda x: x

def parse_args(args):
    print(args)
    gpu = get_free_gpu(8)
    gpu_choice = gpu[0]
    torch.cuda.set_device(int(gpu_choice))
    PATH = args['PATH']
    if not os.path.exists(PATH + 'all_data.pt'):
        process_old_setup(PATH, tensor_name=args['tensor_name'])
        concat_old_side_info(PATH, args['side_info_name'])
    side_info = load_side_info(side_info_path=PATH, indices=args['side_info_order'])
    shape = pickle.load(open(PATH + 'full_tensor_shape.pickle', 'rb'))
    if args['temporal_tag'] is not None:
        for i in args['temporal_tag']:
            side_info[i]['temporal'] = True
    if args['delete_side_info'] is not None:
        for i in args['delete_side_info']:
            del side_info[i]
    if args['kernels'] is None:
        args['kernels'] = ['rbf']#['matern_1', 'matern_2', 'matern_3', 'rbf']
    if args['special_mode']==1:
        for i,v in side_info.items():
            side_info[i]['data'] = torch.ones_like(v['data'])
    elif args['special_mode']==2:
        for i,v in side_info.items():
            side_info[i]['data'] = torch.randn_like(v['data'])
    primal_dims = list(shape)
    for key, val in side_info.items():
        print(key)
        print(val['data'].shape[1])
        primal_dims[key] = val['data'].shape[1]
    print(primal_dims)
    print(f'USING GPU:{gpu_choice}')
    print(shape)
    other_configs = {
        'reg_para_a': args['reg_para_a'],  # Regularization term! Need to choose wisely
        'reg_para_b': args['reg_para_b'],
        'batch_size_a': 1.0 if args['full_grad'] else args['batch_size_a'],
        'batch_size_b': 1.0 if args['full_grad'] else args['batch_size_b'],
        'fp_16': args['fp_16'],  # Wanna use fp_16? Initialize smartly!
        'fused': args['fused'],
        'hyperits': args['hyperits'],
        'save_path': args['save_path'],
        'task': args['task'],
        'epochs': args['epochs'],
        'bayesian': args['bayesian'],  # Mean field does not converge to something meaningful?!
        'data_path': PATH + 'all_data.pt',
        'cuda': args['cuda'],
        'device': f'cuda:{gpu_choice}',
        'train_loss_interval_print': args['sub_epoch_V'] // 2,
        'sub_epoch_V': args['sub_epoch_V'],
        'factorize_latent': args['factorize_latent'],
        'config': {
            'full_grad': args['full_grad'],
            'deep_kernel': args['deep_kernel'],
            'deep': args['deep'],
            'non_lin': get_non_lin(args['non_lin'])
        },
        'shape': shape,
        'architecture': args['architecture'],
        'max_R': args['max_R'],
        'max_lr': args['max_lr'],
        'old_setup': args['old_setup'],
        'latent_scale': args['latent_scale'],
        'chunks': args['chunks'],
        'primal_list': primal_dims,
        'dual': args['dual'],
        'init_max': args['init_max'],
        'L': args['L'],
        'kernels':args['kernels'],
        'multivariate':args['multivariate'],
        'mu_a': args['mu_a'],
        'mu_b': args['mu_b'],
        'sigma_a': args['sigma_a'],
        'sigma_b': args['sigma_b'],
    }
    return side_info,other_configs

def run_job_func(args):
    side_info,other_configs = parse_args(args)
    torch.cuda.empty_cache()
    j = job_object(
        side_info_dict=side_info,
        configs=other_configs,
        seed=args['seed']
    )
    j.run_hyperparam_opt()
    del j
    torch.cuda.empty_cache()
    return 0

def get_loss_func(train_config):
    if train_config['task']=='reg':
        if train_config['bayesian']:
            loss_func = analytical_reconstruction_error_VI
        else:
            loss_func = torch.nn.MSELoss()
    else:
        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=train_config['pos_weight'])
    return loss_func

def get_tensor_architectures(i,shape,primal_dims,R=2,R_scale=1): #Two component tends to overfit?! Really weird!
    TENSOR_ARCHITECTURES = {
        0:{
           0:{'primal_list':[primal_dims[0]],'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':R,'r_1_latent':1,'r_2_latent':R_scale},
           1:{'primal_list':[primal_dims[1]],'ii': [1], 'r_1': R, 'n_list': [shape[1]], 'r_2': R,'r_1_latent':R_scale,'r_2_latent':R_scale}, #Magnitude of kernel sum
           2:{'primal_list':[primal_dims[2]],'ii': [2], 'r_1': R, 'n_list': [shape[2]], 'r_2': 1,'r_1_latent':R_scale,'r_2_latent':1},
           },
        1:{
           0: {'primal_list':[primal_dims[0]],'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':R,'r_1_latent':1,'r_2_latent':R_scale},
           1: {'primal_list':[primal_dims[1],primal_dims[2]],'ii': [1,2], 'r_1': R, 'n_list': [shape[1],shape[2]], 'r_2': 1,'r_1_latent':R_scale,'r_2_latent':1}, #Magnitude of kernel sum
           },
    }
    return TENSOR_ARCHITECTURES[i]

class Log1PlusExp(torch.autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = x.exp()
        ctx.save_for_backward(x)
        y = exp.log1p()
        return x.where(torch.isinf(exp),y.half() if x.type()=='torch.cuda.HalfTensor' else y )

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (-x).exp().half() if x.type()=='torch.cuda.HalfTensor' else (-x).exp()
        return grad_output / (1 + y)

class stableBCEwithlogits(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(stableBCEwithlogits, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.f = Log1PlusExp.apply

    def forward(self, x, y):
        return torch.mean(self.f(x)-x*y)

def analytical_reconstruction_error_VI(y,middle_term,last_term):
    loss = torch.mean(y**2 -2*y*middle_term+last_term)
    return loss

def auc_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(Y.cpu().numpy(), y_pred, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        return auc

def print_garbage():
    obj_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_list.append(obj)
        except:
            pass
    print(len(obj_list))

def opt_32_reinit(model,train_config,lr):
    opt = torch.optim.Adam(model.parameters(), lr=train_config[lr], amsgrad=False)
    return opt

def para_func(df):
    df.sort(axis=1)
    percentiles_list = cal_list + [100-el for el in reversed(cal_list)]
    data = []
    for p in percentiles_list:
        tmp = np.percentile(df,p,axis=1)
        data.append(tmp.reshape(-1,1))
    data = np.concatenate(data,axis=1)
    return data

def calculate_calibration_objective(predictions,indices):
    cal_list_error= []
    for i in range(indices.shape[1]):
        predictions[f'idx_{i}'] = indices[:,i].numpy()
    for cal in cal_list:
        predictions[f'calibrated_{cal}'] = (predictions['y_true'] > predictions[f'{cal}%']) & (predictions['y_true'] < predictions[f'{100-cal}%'])
        _cal_rate = predictions[f'calibrated_{cal}'].sum() / len(predictions)
        print(_cal_rate)
        err = abs((1-2*cal/100)-_cal_rate)
        cal_list_error.append(err)
    total_cal_error = sum(cal_list_error)
    cal_dict = {i: j for i, j in zip(cal_list, cal_list_error)}
    return total_cal_error,cal_dict ,predictions


def para_summary(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    inputs = np.array_split(df,10)
    p = mp.Pool(2)
    results = np.concatenate(p.map(para_func,inputs),axis=0)
    data = np.concatenate([mean.reshape(-1, 1), std.reshape(-1, 1),results], axis=1)
    print(["mean", "std"]+ cal_string_list)
    dataframe = pd.DataFrame(data,columns=["mean", "std"]+ cal_string_list)
    p.close()
    return dataframe


class job_object():
    def __init__(self, side_info_dict, configs, seed):
        """
        :param side_info_dict: Dict containing side info EX) {i:{'data':side_info,'temporal':True}}
        :param tensor_architecture: Tensor architecture  EX) {0:{ii:[0,1],...}
        """
        self.side_info = side_info_dict
        self.hyper_parameters = {}
        self.a = configs['reg_para_a']
        self.b = configs['reg_para_b']
        self.a_ = configs['batch_size_a']
        self.b_ = configs['batch_size_b'] #1.0 max
        self.fp_16 = configs['fp_16']
        self.fused = configs['fused']
        self.hyperits = configs['hyperits']
        self.save_path = configs['save_path']
        self.architecture = configs['architecture']
        self.name = f'bayesian_{seed}' if configs['bayesian'] else f'frequentist_{seed}_architecture_{self.architecture}'
        self.task = configs['task']
        self.epochs = configs['epochs']
        self.bayesian = configs['bayesian']
        self.data_path = configs['data_path']
        self.cuda = configs['cuda']
        self.device = configs['device'] if self.cuda else 'cpu'
        self.train_loss_interval_print  = configs['train_loss_interval_print']
        self.sub_epoch_V = configs['sub_epoch_V']
        self.config = configs['config']
        self.deep_kernel = self.config['deep_kernel']
        self.shape = configs['shape']
        self.max_R = configs['max_R']
        self.max_lr = configs['max_lr']
        self.old_setup = configs['old_setup']
        self.latent_scale = configs['latent_scale']
        self.chunks = configs['chunks']
        self.primal_dims = configs['primal_list']
        self.lrs = [self.max_lr/10**i for i in range(2)]
        self.dual = configs['dual']
        self.max_L = configs['L']
        self.init_range = configs['init_max']
        self.factorize_latent = configs['factorize_latent']
        self.kernels = configs['kernels']
        self.multivariate = configs['multivariate']
        self.mu_a = configs['mu_a']
        self.sigma_a = configs['sigma_a']
        self.mu_b = configs['mu_b']
        self.sigma_b = configs['sigma_b']
        if not self.task=='reg':
            self.pos_weight = configs['pos_weight']
        self.seed = seed
        self.trials = Trials()
        self.define_hyperparameter_space()
        if self.bayesian:
            self.best = np.inf
    def calculate_loss_no_grad(self, task='reg', mode='val'):
        with torch.no_grad():
            loss_list = []
            y_s = []
            _y_preds = []
            self.dataloader.set_mode(mode)
            for i in range(self.dataloader.chunks):
                X, y = self.dataloader.get_chunk(i)
                if self.train_config['cuda']:
                    X = X.to(self.train_config['device'])
                    y = y.to(self.train_config['device'])
                loss, y_pred = self.correct_validation_loss(X, y)
                loss_list.append(loss)
                y_s.append(y.cpu())
                _y_preds.append(y_pred.cpu())
            total_loss = torch.tensor(loss_list).mean().data
            Y = torch.cat(y_s, dim=0)
            y_preds = torch.cat(_y_preds)
            print(y_preds.mean())
            if task == 'reg':
                var_Y = Y.var()
                ref_metric = 1. - total_loss / var_Y
                ref_metric = ref_metric.numpy()
            else:
                ref_metric = auc_check(y_preds, Y)
        return ref_metric

    def train_monitor(self,total_loss, reg, pred_loss, y_pred, p):
        with torch.no_grad():
            ERROR = False
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print('FOUND INF/NAN RIP, RESTARTING')
                print(reg)
                print(pred_loss)
                print(y_pred.mean())
                self.reset_()
                ERROR = True
            if (y_pred == 0).all():
                self.reset_()
            if p % self.train_config['train_loss_interval_print'] == 0:
                print(f'reg_term it {p}: {reg.data}')
                print(f'train_loss it {p}: {pred_loss.data}')
        return ERROR

    def reset_(self):
        fac = self.train_config['reset']
        print(f'dead model_reinit factor: {fac}')
        for n, param in self.model.named_parameters():
            if 'core_param' in n:
                param.normal_(0, self.train_config['reset'])
        self.train_config['reset'] = self.train_config['reset'] * 1.1
        self.train_config['V_lr'] = self.train_config['V_lr']/10

    def correct_validation_loss(self,X, y, ):
        with torch.no_grad():
            if self.train_config['task'] == 'reg':
                loss_func = torch.nn.MSELoss()
                if self.train_config['bayesian']:
                    y_pred, _, _ = self.model(X)
                else:
                    y_pred, _ = self.model(X)
                pred_loss = loss_func(y_pred, y.squeeze())
            else:
                loss_func = torch.nn.BCELoss()
                y_pred, reg = self.model(X)
                pred_loss = loss_func(y_pred, y.squeeze())
            return pred_loss, y_pred

    def correct_forward_loss(self,X, y, loss_func):
        if self.train_config['task'] == 'reg' and self.train_config['bayesian']:
            y_pred, last_term, reg = self.model(X)
            pred_loss = loss_func(y.squeeze(), y_pred, last_term)
            total_loss = pred_loss*self.train_config['sigma_y'] + reg
        else:
            y_pred, reg = self.model(X)
            pred_loss = loss_func(y_pred, y.squeeze())
            total_loss = pred_loss + reg
        return total_loss, reg, pred_loss, y_pred


    def train_loop(self, opt, loss_func, warmup=False):
        sub_epoch = self.train_config['sub_epoch_V']
        lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=sub_epoch//2, factor=0.9)
        print_garbage()
        torch.cuda.empty_cache()
        for p in range(sub_epoch + 1):
            if warmup:
                self.dataloader.ratio = self.train_config['batch_ratio'] / 1e2
            else:
                self.dataloader.ratio = self.train_config['batch_ratio']
            self.dataloader.set_mode('train')
            X, y = self.dataloader.get_batch()
            if self.train_config['cuda']:
                X = X.to(self.train_config['device'])
                y = y.to(self.train_config['device'])
            start = time.time()
            total_loss, reg, pred_loss, y_pred = self.correct_forward_loss(X, y,loss_func)
            end = time.time()
            print(f'forward time: {end-start}')
            opt.zero_grad()
            start = time.time()
            if self.train_config['fp_16'] and not warmup:
                with self.train_config['amp'].scale_loss(total_loss, opt, loss_id=0) as loss_scaled:
                    loss_scaled.backward()
            else:
                total_loss.backward()
            opt.step()
            end = time.time()
            print(f'backward time: {end-start}')


            lrs.step(total_loss)
            if p % (sub_epoch // 2) == 0:
                torch.cuda.empty_cache()
                l_val = self.calculate_loss_no_grad(mode='val',task=self.train_config['task'])
                torch.cuda.empty_cache()
                l_test = self.calculate_loss_no_grad(mode='test',task=self.train_config['task'])
                torch.cuda.empty_cache()
                print(f'val_error= {l_val}')
                print(f'test_error= {l_test}')
                if self.bayesian:
                    self.model.get_norms()
            ERROR = self.train_monitor(total_loss, reg, pred_loss, y_pred,  p)
            if ERROR:
                return ERROR
        del lrs
        return False

    def outer_train_loop(self, opts, loss_func, ERROR, train_list, train_dict, warmup=False,toggle=False):
        for i in train_list:
            print_garbage()
            settings = train_dict[i]
            f = settings['call']
            lr = settings['para']
            f()
            print(lr)
            opt = opts[lr]
            if self.bayesian:
                self.model.toggle(toggle)
                if not toggle and lr=='ls_lr':
                    continue
            ERROR = self.train_loop(opt, loss_func, warmup=warmup)
            print_garbage()
            if ERROR:
                return ERROR
            print(torch.cuda.memory_cached() / 1e6)
            print(torch.cuda.memory_allocated() / 1e6)
            torch.cuda.empty_cache()
        return ERROR

    def opt_reinit(self, lr_params, warmup=False):
        if self.train_config['fp_16'] and not warmup:
            import apex
            from apex import amp
            amp.register_float_function(torch, 'bmm')  # TODO: HALLELUJA
            if self.train_config['fused']:
                tmp_opt_list = []
                for lr in lr_params:
                    tmp_opt_list.append(apex.optimizers.FusedAdam(self.model.parameters(), lr=self.train_config[
                        lr]))  # Calling this again makes the model completely in FP16 wtf
                [self.model], tmp_opt_list = amp.initialize([self.model], tmp_opt_list, opt_level='O1', num_losses=1)
                opts = {x: y for x, y in zip(lr_params, tmp_opt_list)}
                self.model.amp = amp
                self.model.amp_patch(amp)
            else:
                tmp_opt_list = []
                for lr in lr_params:
                    tmp_opt_list.append(torch.optim.Adam(self.model.parameters(), lr=self.train_config[lr], amsgrad=False))
                [self.model], tmp_opt_list = amp.initialize([self.model], tmp_opt_list, opt_level='O1', num_losses=1)
                opts = {x: y for x, y in zip(lr_params, tmp_opt_list)}
                [self.model], tmp_opt_list = amp.initialize([self.model], tmp_opt_list, opt_level='O1', num_losses=1)
                self.model.amp = amp
                self.model.amp_patch(amp)
            self.train_config['amp'] = amp
        else:
            tmp_opt_list = []
            for lr in lr_params:
                tmp_opt_list.append(torch.optim.Adam(self.model.parameters(), lr=self.train_config[lr], amsgrad=False))
            opts = {x: y for x, y in zip(lr_params, tmp_opt_list)}
        return opts

    def setup_runs(self, warmup):
        loss_func = get_loss_func(self.train_config)
        ERROR = False
        kernel, deep_kernel = self.model.has_kernel_component()
        train_dict = {0: {'para': 'V_lr', 'call': self.model.turn_on_V}}
        train_list = [0]
        if not self.train_config['old_setup']:
            train_dict[1] = {'para': 'prime_lr', 'call': self.model.turn_on_prime}
            train_list.append(1)
        if self.train_config['dual']:
            if kernel:
                train_dict[2] = {'para': 'ls_lr', 'call': self.model.turn_on_kernel_mode}
                train_list.insert(-1, 2)
            if deep_kernel:
                train_dict[3] = {'para': 'deep_lr', 'call': self.model.turn_on_deep_kernel}
                train_list.insert(-2, 3)
        if warmup:
            train_list = [0]
            if not self.train_config['old_setup']:
                train_list.append(1)
        lrs = [v['para'] for v in train_dict.values()]
        opts = self.opt_reinit( lrs, warmup=warmup)
        return  opts, loss_func, ERROR, train_list, train_dict

    def train(self,toggle=False):
        self.train_config['reset'] = 1.0
        opts,loss_func,ERROR,train_list,train_dict = self.setup_runs(warmup=False)
        for i in range(self.train_config['epochs']):
            ERROR = self.outer_train_loop(opts,loss_func,ERROR,train_list,train_dict, warmup=False,toggle=toggle)
            if ERROR:
                if self.bayesian:
                    return -np.inf, -np.inf,-np.inf, -np.inf,-np.inf, -np.inf,-np.inf
                else:
                    return -np.inf, -np.inf
        val_loss_final = self.calculate_loss_no_grad(mode='val',task=self.train_config['task'])
        test_loss_final = self.calculate_loss_no_grad(mode='test',task=self.train_config['task'])
        del opts
        if self.bayesian:
            total_cal_error_val,val_cal_dict ,_ = self.calculate_calibration(mode='val',task=self.train_config['task'])
            total_cal_error_test,test_cal_dict ,predictions = self.calculate_calibration(mode='test',task=self.train_config['task'])
            print(val_loss_final,test_loss_final)
            print(total_cal_error_val,total_cal_error_test)
            print(val_cal_dict)
            print(test_cal_dict)
            return total_cal_error_val-val_loss_final,total_cal_error_test-test_loss_final,val_cal_dict,test_cal_dict,val_loss_final,test_loss_final,predictions
        else:
            return val_loss_final,test_loss_final

    def define_hyperparameter_space(self):
        self.hyperparameter_space = {}
        self.available_side_info_dims = []
        t_act = get_tensor_architectures(self.architecture,self.shape,self.primal_dims, 2)
        for dim, val in self.side_info.items():
            self.available_side_info_dims.append(dim)
            if self.dual:
                if self.fp_16:
                    self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', ['rbf'])
                else:
                    self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', self.kernels)
                self.hyperparameter_space[f'ARD_{dim}'] = hp.choice(f'ARD_{dim}', [True, False])

        if self.bayesian:
            self.hyperparameter_space['reg_para'] = hp.uniform('reg_para', self.a, self.b)
            for i in range(len(t_act)):
                if self.latent_scale:
                    self.hyperparameter_space[f'mu_prior_s_{i}'] = hp.uniform(f'mu_prior_s_{i}', self.mu_a,
                                                                                  self.mu_b)
                    self.hyperparameter_space[f'sigma_prior_s_{i}'] = hp.uniform(f'sigma_prior_s_{i}',
                                                                                     self.sigma_a, self.sigma_b)
                    self.hyperparameter_space[f'mu_prior_b_{i}'] = hp.uniform(f'mu_prior_b_{i}', self.mu_a,
                                                                                  self.mu_b)
                    self.hyperparameter_space[f'sigma_prior_b_{i}'] = hp.uniform(f'sigma_prior_b_{i}',
                                                                                     self.sigma_a, self.sigma_b)
                else:
                    self.hyperparameter_space[f'mu_prior_prime_{i}'] = hp.uniform(f'mu_prior_prime_{i}', self.mu_a, self.mu_b)
                    self.hyperparameter_space[f'sigma_prior_prime_{i}'] = hp.uniform(f'sigma_prior_prime_{i}', self.sigma_a, self.sigma_b)
                self.hyperparameter_space[f'mu_prior_{i}'] = hp.uniform(f'mu_prior_{i}', self.mu_a, self.mu_b)
                if not self.multivariate or not i in self.available_side_info_dims:
                    self.hyperparameter_space[f'sigma_prior_{i}'] = hp.uniform(f'sigma_prior_{i}', self.sigma_a,
                                                                               self.sigma_b)
        else:
            for i in range(len(t_act)):
                self.hyperparameter_space[f'reg_para_{i}'] = hp.uniform(f'reg_para_{i}', self.a, self.b)
                if self.latent_scale:
                    self.hyperparameter_space[f'reg_para_s_{i}'] = hp.uniform(f'reg_para_s_{i}', self.a, self.b)
                    self.hyperparameter_space[f'reg_para_b_{i}'] = hp.uniform(f'reg_para_b_{i}', self.a, self.b)
                if not self.old_setup:
                    if self.factorize_latent:
                        self.hyperparameter_space[f'prime_{i}'] = hp.choice(f'prime_{i}', [False,True])
                        self.hyperparameter_space['sub_R'] = hp.choice('sub_R',
                                                                       np.arange(self.max_R // 4, self.max_R // 2,
                                                                                 dtype=int))
                    if not self.dual:
                        self.hyperparameter_space[f'reg_para_prime_{i}'] = hp.uniform(f'reg_para_prime_{i}', self.a, self.b)


        if self.config['deep']:
            self.hyperparameter_space['L'] = hp.choice('L', np.arange(1,self.max_L,dtype=int))
        self.hyperparameter_space['batch_size_ratio'] = hp.uniform('batch_size_ratio', self.a_, self.b_)
        if self.latent_scale:
            self.hyperparameter_space['R_scale'] = hp.choice('R_scale', np.arange(self.max_R//4,self.max_R//2+1,dtype=int))
        self.hyperparameter_space['R'] = hp.choice('R', np.arange( int(round(self.max_R*0.8)),self.max_R+1,dtype=int))
        self.hyperparameter_space['lr_2'] = hp.choice('lr_2', self.lrs ) #Very important for convergence

    def init_and_train(self,parameters):
        self.config['dual'] = self.dual
        self.tensor_architecture = get_tensor_architectures(self.architecture, self.shape,self.primal_dims, parameters['R'],parameters['R_scale'] if self.latent_scale else 1)
        if not self.old_setup:
            if not self.latent_scale:
                for key, component in self.tensor_architecture.items():
                    if self.factorize_latent:
                        component['prime'] = parameters[f'prime_{key}']
                        self.config['sub_R'] = parameters['sub_R']
                    else:
                        self.config['sub_R'] = 1
                        component['prime'] = False
        if self.config['deep']:
            self.config['L'] = parameters['L']
        lambdas = self.extract_reg_terms(parameters)
        init_dict = self.construct_init_dict(parameters)
        self.train_config = self.extract_training_params(parameters)
        print(parameters)
        if self.bayesian:
            if self.latent_scale:
                self.model = varitional_KFT_scale(initialization_data=init_dict,
                                            cuda=self.device, config=self.config, old_setup=self.old_setup,lambdas=lambdas)
            else:
                self.model = variational_KFT(initialization_data=init_dict,
                                            cuda=self.device, config=self.config, old_setup=self.old_setup,lambdas=lambdas)
        else:
            if self.latent_scale:
                self.model = KFT_scale(initialization_data=init_dict, cuda=self.device,
                            config=self.config, old_setup=self.old_setup,lambdas=lambdas)
            else:
                self.model = KFT(initialization_data=init_dict, cuda=self.device,
                            config=self.config, old_setup=self.old_setup,lambdas=lambdas)
        if self.cuda:
            self.model = self.model.to(self.device)
        print(self.model)
        self.dataloader = get_dataloader_tensor(self.data_path, seed=self.seed, mode='train',
                                                 bs_ratio=parameters['batch_size_ratio'])
        self.dataloader.chunks = self.train_config['chunks']
        torch.cuda.empty_cache()
        if self.bayesian:
            self.train(toggle=True)
            print('sigma train')
            total_cal_error_val,total_cal_error_test,val_cal_dict,test_cal_dict,val_loss_final,test_loss_final,predictions = self.train(toggle=False)
            del self.model
            del self.dataloader
            torch.cuda.empty_cache()
            return total_cal_error_val,total_cal_error_test,val_cal_dict,test_cal_dict,val_loss_final,test_loss_final,predictions

        else:
            val_loss_final, test_loss_final = self.train()
            del self.model
            del self.dataloader
            torch.cuda.empty_cache()
            return val_loss_final, test_loss_final

    def calculate_calibration(self,mode='val',task='reg',samples=100):
        self.dataloader.set_mode(mode)
        with torch.no_grad():
            all_samples = []
            for i in range(samples):
                _y_preds = []
                for i in range(self.dataloader.chunks):
                    X, y = self.dataloader.get_chunk(i)
                    if self.train_config['cuda']:
                        X = X.to(self.train_config['device'])
                    _y_pred_sample = self.model.sample(X)
                    if not task=='reg':
                        _y_pred_sample = torch.sigmoid(_y_pred_sample)
                    _y_preds.append(_y_pred_sample.cpu().numpy())
                y_sample = np.concatenate(_y_preds,axis=0)
                all_samples.append(y_sample)
            Y_preds = np.stack(all_samples,axis=1)
            print(Y_preds.shape)
            df = para_summary(Y_preds)
            df['y_true'] = self.dataloader.Y.numpy()
            total_cal_error,cal_dict ,predictions  =  calculate_calibration_objective(df,self.dataloader.X)
            return total_cal_error,cal_dict ,predictions
    def __call__(self, parameters):
        for i in range(2):
        # try:
        #     pykeops.clean_pykeops()  # just in case old build files are still present
            torch.cuda.empty_cache()
            get_free_gpu(10)  # should be 0 between calls..
            if self.bayesian:
                total_cal_error_val,total_cal_error_test,val_cal_dict,test_cal_dict,val_loss_final,test_loss_final,predictions = self.init_and_train(parameters)
                if not np.isinf(val_loss_final):
                    torch.cuda.empty_cache()
                    if total_cal_error_test < self.best:
                        self.best = total_cal_error_test
                        predictions.to_parquet(self.save_path + '/'+f'VI_predictions_{self.seed}', engine='fastparquet')
                    return {'loss': total_cal_error_val,
                            'status': STATUS_OK,
                            'test_loss': total_cal_error_test,
                            'val_cal_dict':val_cal_dict,
                            'test_cal_dict':test_cal_dict,
                            'val_loss_final':val_loss_final,
                            'test_loss_final':test_loss_final}
            else:
                val_loss_final, test_loss_final = self.init_and_train(parameters)
                if not np.isinf(val_loss_final):
                    torch.cuda.empty_cache()
                    return {'loss': -val_loss_final, 'status': STATUS_OK, 'test_loss': -test_loss_final}
            # except Exception as e:
            # print(e)
            torch.cuda.empty_cache()
        return {'loss': np.inf, 'status': STATUS_FAIL, 'test_loss': np.inf}

    def get_kernel_vals(self,desc):
        if 'matern_1'== desc:
            return 'matern',0.5
        elif 'matern_2' == desc:
            return 'matern',1.5
        elif 'matern_3' == desc:
            return 'matern',2.5
        else:
            return desc,None

    def extract_reg_terms(self, parameters):
        reg_params = dict()
        if self.bayesian:
            for i in range(len(self.tensor_architecture)):
                reg_params[f'reg_para_prime_{i}'] = 1.0
                reg_params[f'reg_para_{i}'] = 1.0
                reg_params[f'reg_para_s_{i}'] = 1.0
                reg_params[f'reg_para_b_{i}'] = 1.0
        else:
            for i in range(len(self.tensor_architecture)):
                reg_params[f'reg_para_{i}'] = parameters[f'reg_para_{i}']
                if self.latent_scale:
                    reg_params[f'reg_para_s_{i}'] = parameters[f'reg_para_s_{i}']
                    reg_params[f'reg_para_b_{i}'] = parameters[f'reg_para_b_{i}']
                if not self.old_setup and not self.dual:
                    reg_params[f'reg_para_prime_{i}'] = parameters[f'reg_para_prime_{i}']
                else:
                    reg_params[f'reg_para_prime_{i}'] = 1.0
        print(reg_params)
        return reg_params

    def construct_kernel_params(self,side_info_dims,parameters):
        kernel_param = {}
        for i in range(len(side_info_dims)):
            dim = side_info_dims[i]
            if dim in self.available_side_info_dims:
                k,nu = self.get_kernel_vals(parameters[f'kernel_{dim}_choice'])
                kernel_param[i+1] = {'ARD':parameters[f'ARD_{dim}'],'ls_factor':1.0,'nu':nu,'kernel_type':k,'p':1.0}
        return kernel_param

    def construct_side_info_params(self,side_info_dims):
        side_params = {}
        for i in range(len(side_info_dims)):
            dim = side_info_dims[i]
            if dim in self.available_side_info_dims:
                side_params[i+1] = self.side_info[dim]['data']
        return side_params

    def construct_init_dict(self,parameters):
        init_dict = self.tensor_architecture
        for key,items in init_dict.items():
            side_info_dims = items['ii']
            if self.dual:
                kernel_param  = self.construct_kernel_params(side_info_dims,parameters)
            else:
                kernel_param = {}
            side_param = self.construct_side_info_params(side_info_dims)
            items['kernel_para'] = kernel_param
            items['side_info'] = side_param
            if side_param:
                items['has_side_info'] = True
            else:
                items['has_side_info'] = False
            items['init_scale'] = self.init_range
            if self.bayesian:
                items['multivariate'] = self.multivariate
                if self.latent_scale:
                    items['mu_prior_s'] = parameters[f'mu_prior_s_{key}']
                    items['sigma_prior_s'] = parameters[f'sigma_prior_s_{key}']
                    items['mu_prior_b'] = parameters[f'mu_prior_b_{key}']
                    items['sigma_prior_b'] = parameters[f'sigma_prior_b_{key}']
                else:
                    items['mu_prior_prime'] = parameters[f'mu_prior_prime_{key}']
                    items['sigma_prior_prime'] = parameters[f'sigma_prior_prime_{key}']
                items['mu_prior'] = parameters[f'mu_prior_{key}']
                if not self.multivariate or not key in self.available_side_info_dims:
                    items['sigma_prior'] = parameters[f'sigma_prior_{key}']


        return init_dict

    def extract_training_params(self,parameters):
        training_params = {}
        training_params['fp_16'] = (self.fp_16 and not self.bayesian)
        training_params['fused'] = self.fused
        training_params['task'] = self.task
        training_params['epochs'] = self.epochs
        training_params['prime_lr'] = parameters['lr_2']/100.
        training_params['V_lr'] = parameters['lr_2']
        training_params['ls_lr'] = parameters['lr_2']/100.
        training_params['device'] = self.device
        training_params['cuda'] = self.cuda
        training_params['train_loss_interval_print']=self.train_loss_interval_print
        training_params['sub_epoch_V']=self.sub_epoch_V
        training_params['bayesian'] = self.bayesian
        training_params['old_setup'] = self.old_setup
        training_params['architecture'] = self.architecture
        training_params['deep_kernel'] = self.deep_kernel
        training_params['batch_ratio'] = parameters['batch_size_ratio']
        training_params['chunks'] = self.chunks
        training_params['dual'] = self.dual
        if not self.task=='reg':
            training_params['pos_weight'] = self.pos_weight

        if self.bayesian:
            training_params['sigma_y'] = parameters['reg_para']
        if self.deep_kernel:
            training_params['deep_lr'] = 1e-3#parameters['lr_4']
        return training_params

    def run_hyperparam_opt(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        best = fmin(fn=self,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.hyperits,
                    trials=self.trials,
                    verbose=1)
        print(space_eval(self.hyperparameter_space, best))
        pickle.dump(self.trials,
                    open(self.save_path +'/'+ self.name + '.p',
                         "wb"))

