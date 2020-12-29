import torch
from KFT.KFT import KFT, variational_KFT,KFT_scale,varitional_KFT_scale
from torch.nn.modules.loss import _Loss
import gc
import pickle
import os
import warnings
from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK,STATUS_FAIL
from KFT.util import *
from sklearn import metrics
import time
import numpy as np
import multiprocessing as mp
import pandas as pd
from torch.cuda.amp import autocast,GradScaler


cal_list = [5, 15, 25, 35, 45]
cal_string_list = [f'{cal}%' for cal in cal_list] + [f'{100-cal}%'for cal in reversed(cal_list)]


def run_job_func(args):
    torch.cuda.empty_cache()
    j = job_object(args)
    j.run_hyperparam_opt()

def get_loss_func(train_config):
    if train_config['task']=='regression':
        if train_config['bayesian']:
            loss_func = analytical_reconstruction_error_VI
        else:
            loss_func = torch.nn.MSELoss()
    else:
        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=train_config['pos_weight'])
    return loss_func

def get_tensor_architectures(i, permuted_shape, R=2, R_scale=1): #Two component tends to overfit?! Really weird!

    if len(permuted_shape)==3:
        TENSOR_ARCHITECTURES = {
            0:{
               0:{'ii':[0],'r_1':1,'n_list':[permuted_shape[0]], 'r_2':R, 'r_1_latent':1, 'r_2_latent':R_scale},
               1:{'ii': [1], 'r_1': R, 'n_list': [permuted_shape[1]], 'r_2': R, 'r_1_latent':R_scale, 'r_2_latent':R_scale}, #Magnitude of kernel sum
               2:{'ii': [2], 'r_1': R, 'n_list': [permuted_shape[2]], 'r_2': 1, 'r_1_latent':R_scale, 'r_2_latent':1},
               },
            1:{
               0: {'ii':[0],'r_1':1,'n_list':[permuted_shape[0]], 'r_2':R, 'r_1_latent':1, 'r_2_latent':R_scale},
               1: {'ii': [1,2], 'r_1': R, 'n_list': [permuted_shape[1], permuted_shape[2]], 'r_2': 1, 'r_1_latent':R_scale, 'r_2_latent':1}, #Magnitude of kernel sum
               },
            2: {
                0: {'ii': [0], 'r_1': 1, 'n_list': [permuted_shape[0]], 'r_2': R,
                    'r_1_latent': 1, 'r_2_latent': R_scale},
                1: {'ii': [1], 'r_1': R, 'n_list': [permuted_shape[1]], 'r_2': 1,
                    'r_1_latent': R_scale, 'r_2_latent': 1},  # Magnitude of kernel sum
            }  # Regular MF

        }
    elif len(permuted_shape)==2:
        TENSOR_ARCHITECTURES = {

            0: {
                0: { 'ii': [0], 'r_1': 1, 'n_list': [permuted_shape[0]], 'r_2': R,
                    'r_1_latent': 1, 'r_2_latent': R_scale},
                1: {'ii': [1], 'r_1': R, 'n_list': [permuted_shape[1]], 'r_2': 1,
                    'r_1_latent': R_scale, 'r_2_latent': 1},  # Magnitude of kernel sum
            }  # Regular MF
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

def accuracy_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        return np.mean(y_pred==Y.cpu().numpy())

def print_garbage():
    obj_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_list.append(obj)
        except:
            pass
    print(len(obj_list))


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
    def __init__(self, args):
        """
        :param side_info_dict: Dict containing side info EX) {i:{'data':side_info,'temporal':True}}
        :param tensor_architecture: Tensor architecture  EX) {0:{ii:[0,1],...}
        """

        self.tensor_component_configs, args, self.side_info = self.process_input(args)
        for key,val in args.items():
            setattr(self,key,val)

        self.device = args['device'] if self.cuda else 'cpu'
        self.lrs = [self.max_lr/10**i for i in range(2)]
        self.name = f'bayesian_{self.seed}' if args['bayesian'] else f'frequentist_{self.seed}'
        if not self.task=='regression':
            self.pos_weight =torch.tensor(args['pos_weight']).to(self.device) if self.cuda else torch.tensor(args['pos_weight'])
        self.trials = Trials()
        self.hyper_parameters = {}
        self.define_hyperparameter_space()
        if self.bayesian:
            self.best = np.inf

    def process_input(self,args):
        print(args)
        devices = GPUtil.getAvailable(order='memory', limit=1)
        device = devices[0]
        torch.cuda.set_device(int(device))
        PATH = args['PATH']
        original_shape = list(pickle.load(open(PATH + 'full_tensor_shape.pickle', 'rb')))
        loaded_side_info = list(load_side_info(side_info_path=PATH+'side_info.pt', shape=original_shape))
        side_info_dict_tmp = {} #'dim_idx':{'side_info_for_that_dim':0,'temporal':False}
        for dim_idx,n in enumerate(original_shape):
            for idx,s in enumerate(loaded_side_info):
                if s.shape[0]==n:
                    side_info_dict_tmp[dim_idx]={'data':loaded_side_info.pop(idx),'temporal_tag':False}
        if args['temporal_tag'] is not None:
            side_info_dict_tmp[args['temporal_tag']]['temporal_tag'] = True
        if args['special_mode'] == 1:
            for i, v in side_info_dict_tmp.items():
                side_info_dict_tmp[i]['data'] = torch.ones_like(v['data']) #relative to the orginal shape
        elif args['special_mode'] == 2:
            for i, v in side_info_dict_tmp.items():
                side_info_dict_tmp[i]['data'] = torch.randn_like(v['data']) #relative to the orginal shape
        if args['delete_side_info'] is not None: # a list if passed to the orginal shape
            for i in args['delete_side_info']:
                del side_info_dict_tmp[i]

        for dim_idx, n in enumerate(original_shape):
            if dim_idx in side_info_dict_tmp:
                if not args['dual']:
                    original_shape[dim_idx] = side_info_dict_tmp[dim_idx]['data'].shape[1]

        side_info_dict = {}
        for perm,old_key in zip(args['shape_permutation'],[idx for idx in range(len(original_shape))]):
            if perm in side_info_dict_tmp:
                side_info_dict[old_key] = side_info_dict_tmp.pop(perm)
        shape = [original_shape[el] for el in args['shape_permutation']]
        args['kernels'] = ['matern_1', 'matern_2', 'matern_3', 'rbf']  # ['matern_1', 'matern_2', 'matern_3', 'rbf']
        print(f'USING GPU:{device}')
        args['batch_size_a'] = 1.0 if args['full_grad'] else args['batch_size_a']
        args['batch_size_b'] = 1.0 if args['full_grad'] else args['batch_size_b']
        args['data_path'] = PATH + 'all_data.pt'
        args['device'] = f'cuda:{device}'
        args['train_loss_interval_print'] = args['sub_epoch_V'] // 2
        args['shape'] = shape
        print(shape)
        tensor_component_configs = {
                             'full_grad': args['full_grad'],
                             'bayesian': args['bayesian'],
                         }
        return tensor_component_configs,args,side_info_dict


    def calculate_loss_no_grad(self, task='reg', mode='val',final=False):
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
                loss, y_pred = self.correct_validation_loss(X, y,final)
                loss_list.append(loss)
                y_s.append(y.cpu())
                _y_preds.append(y_pred.cpu())
            total_loss = torch.tensor(loss_list).mean().item()
            Y = torch.cat(y_s, dim=0)
            y_preds = torch.cat(_y_preds)
            print(f'{mode} loss_func_loss: {total_loss}' )
            if task == 'regression':
                if self.bayesian and not final:
                    if self.train_means:
                        var_Y = Y.var()
                        ref_metric = 1. - total_loss / var_Y
                        ref_metric = ref_metric.numpy()
                    else:
                        ref_metric = total_loss
                else:
                    var_Y = Y.var()
                    ref_metric = 1. - total_loss / var_Y
                    ref_metric = ref_metric.numpy()
                    print(f'{mode} NRSME: {total_loss**0.5/Y.abs().mean()}')

            else:
                y_preds = torch.sigmoid(y_preds)
                if task=='classification_auc':
                    ref_metric = auc_check(y_preds, Y)
                elif task=='classification_acc':
                    ref_metric = accuracy_check(y_preds, Y)
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

    def correct_validation_loss(self,X, y,final=False ):
        with torch.no_grad():
            if self.train_config['task'] == 'regression':
                if self.train_config['bayesian']:
                    y_pred, middle_term, reg = self.model(X)
                    if self.train_means or final:
                        loss_func = torch.nn.MSELoss()
                        pred_loss = loss_func(y_pred, y.squeeze())
                    else:
                        pred_loss= -(analytical_reconstruction_error_VI(y.squeeze(),y_pred,middle_term)*self.train_config['sigma_y'] + reg)
                else:
                    loss_func = torch.nn.MSELoss()
                    y_pred, _ = self.model(X)
                    pred_loss = loss_func(y_pred, y.squeeze())
            else:
                loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
                y_pred, reg = self.model(X)
                pred_loss = loss_func(y_pred, y)
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


    def train_loop(self, opt, loss_func):
        sub_epoch = self.train_config['sub_epoch_V']
        lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=sub_epoch//2, factor=0.9)
        torch.cuda.empty_cache()
        for p in range(sub_epoch + 1):
            self.dataloader.ratio = self.train_config['batch_ratio']
            self.dataloader.set_mode('train')
            X, y = self.dataloader.get_batch()
            if self.train_config['cuda']:
                X = X.to(self.train_config['device'])
                y = y.to(self.train_config['device'])
            # start = time.time()
            total_loss, reg, pred_loss, y_pred = self.correct_forward_loss(X, y,loss_func)
            # end = time.time()
            # print(f'forward time: {end-start}')
            opt.zero_grad()
            # start = time.time()
            total_loss.backward()
            opt.step()
            # end = time.time()
            # print(f'backward time: {end-start}')

            lrs.step(total_loss)
            if p % (sub_epoch // 2) == 0:
                # print('period: ',self.model.TT_cores['2'].kernel_1.period_length)
                # print('ls: ',self.model.TT_cores['2'].kernel_1.raw_lengthscale)

                torch.cuda.empty_cache()
                l_val = self.calculate_loss_no_grad(mode='val',task=self.train_config['task'])
                torch.cuda.empty_cache()
                l_test = self.calculate_loss_no_grad(mode='test',task=self.train_config['task'])
                torch.cuda.empty_cache()
                print(f'val_error= {l_val}')
                print(f'test_error= {l_test}')
                if self.kill_counter==100:
                    print(f"No improvement in val, stopping training! best val error: {self.best_val_loss}")
                    return 'early_stop'
                if -l_val < -self.best_val_loss:
                    self.kill_counter = 0
                    self.best_val_loss = l_val
                    self.dump_model(val_loss=l_val, test_loss=l_test, i=self.hyperits_i, parameters=self.parameters) #think carefully here!
                else:
                    self.kill_counter+=1
            ERROR = self.train_monitor(total_loss, reg, pred_loss, y_pred,  p)
            if ERROR:
                return ERROR
        return False

    def outer_train_loop(self, opts, loss_func, ERROR, train_list, train_dict, train_means=False):
        for i in train_list:
            settings = train_dict[i]
            f = settings['call']
            lr = settings['para']
            opt = opts[lr]
            for key,items in self.model.ii.items():
                f(key)
                if self.bayesian:
                    self.model.toggle(train_means)
                    if not train_means and lr== 'ls_lr':
                        continue
                ERROR = self.train_loop(opt, loss_func)

                if ERROR=='early_stop':
                    return ERROR
                if ERROR:
                    return ERROR

                # print(torch.cuda.memory_cached() / 1e6)
                # print(torch.cuda.memory_allocated() / 1e6)
                torch.cuda.empty_cache()
        return ERROR

    def opt_reinit(self, lr_params):

        tmp_opt_list = []
        for lr in lr_params:
            tmp_opt_list.append(torch.optim.Adam(self.model.parameters(), lr=self.train_config[lr], amsgrad=False))
        opts = {x: y for x, y in zip(lr_params, tmp_opt_list)}
        return opts

    def setup_runs(self):
        loss_func = get_loss_func(self.train_config)
        ERROR = False
        kernel = self.model.has_kernel_component()
        train_dict = {0: {'para': 'V_lr', 'call': self.model.turn_on_V}}
        train_list = [0]
        if not self.train_config['old_setup']:
            train_dict[1] = {'para': 'prime_lr', 'call': self.model.turn_on_prime}
            train_list.append(1)
        if self.train_config['dual']:
            if kernel:
                train_dict[2] = {'para': 'ls_lr', 'call': self.model.turn_on_kernel_mode}
                train_list.insert(-1, 2)

        lrs = [v['para'] for v in train_dict.values()]
        opts = self.opt_reinit( lrs)
        return  opts, loss_func, ERROR, train_list, train_dict

    def train(self, train_means=False):
        self.kill_counter = 0
        self.train_config['reset'] = 1.0
        self.train_means = train_means
        self.best_val_loss = -np.inf

        opts,loss_func,ERROR,train_list,train_dict = self.setup_runs()
        for i in range(self.train_config['epochs']):
            ERROR = self.outer_train_loop(opts, loss_func, ERROR, train_list, train_dict, train_means=train_means)
            if ERROR=='early_stop':
                break
            if ERROR:
                if self.bayesian:
                    return -np.inf, -np.inf,-np.inf, -np.inf,-np.inf, -np.inf,-np.inf
                else:
                    return -np.inf, -np.inf
        self.load_dumped_model(self.hyperits_i)
        if self.bayesian:
            val_loss_final = self.calculate_loss_no_grad(mode='val', task=self.train_config['task'],final=True)
            test_loss_final = self.calculate_loss_no_grad(mode='test', task=self.train_config['task'],final=True)
            total_cal_error_val,val_cal_dict ,_ = self.calculate_calibration(mode='val',task=self.train_config['task'])
            total_cal_error_test,test_cal_dict ,predictions = self.calculate_calibration(mode='test',task=self.train_config['task'])
            return total_cal_error_val-val_loss_final,total_cal_error_test-test_loss_final,val_cal_dict,test_cal_dict,val_loss_final,test_loss_final,predictions
        else:
            val_loss_final = self.calculate_loss_no_grad(mode='val', task=self.train_config['task'],final=True)
            test_loss_final = self.calculate_loss_no_grad(mode='test', task=self.train_config['task'],final=True)
            return val_loss_final,test_loss_final

    def define_hyperparameter_space(self):
        self.hyperparameter_space = {}
        self.available_side_info_dims = []
        t_act = get_tensor_architectures(self.architecture, self.shape)
        for dim, val in self.side_info.items():
            self.available_side_info_dims.append(dim)
            if self.dual: #Add periodic kernel setup here
                if val['temporal_tag']:
                    self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice',['periodic'] ) #self.kernels
                else:
                    self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice',self.kernels ) #self.kernels
                self.hyperparameter_space[f'ARD_{dim}'] = hp.choice(f'ARD_{dim}', [True,False])

        if self.bayesian:
            self.hyperparameter_space['reg_para'] = hp.uniform('reg_para', self.reg_para_a, self.reg_para_b)
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
                self.hyperparameter_space[f'reg_para_{i}'] = hp.uniform(f'reg_para_{i}', self.reg_para_a, self.reg_para_b)
                if self.latent_scale:
                    self.hyperparameter_space[f'reg_para_s_{i}'] = hp.uniform(f'reg_para_s_{i}', self.reg_para_a, self.reg_para_b)
                    self.hyperparameter_space[f'reg_para_b_{i}'] = hp.uniform(f'reg_para_b_{i}', self.reg_para_a, self.reg_para_b)
                if not self.old_setup:

                    if not self.dual:
                        self.hyperparameter_space[f'reg_para_prime_{i}'] = hp.uniform(f'reg_para_prime_{i}', self.reg_para_a, self.reg_para_b)

        self.hyperparameter_space['batch_size_ratio'] = hp.uniform('batch_size_ratio', self.batch_size_a, self.batch_size_b)
        if self.latent_scale:
            self.hyperparameter_space['R_scale'] = hp.choice('R_scale', np.arange(self.max_R//4,self.max_R//2+1,dtype=int))
        self.hyperparameter_space['R'] = hp.choice('R', np.arange( int(round(self.max_R*0.5)),self.max_R+1,dtype=int))
        self.hyperparameter_space['lr_2'] = hp.choice('lr_2', self.lrs ) #Very important for convergence

    def init(self, parameters):
        print(parameters)
        self.tensor_component_configs['dual'] = self.dual
        self.tensor_architecture = get_tensor_architectures(self.architecture, self.shape, parameters['R'], parameters['R_scale'] if self.latent_scale else 1)
        if not self.old_setup:
            if not self.latent_scale:
                for key, component in self.tensor_architecture.items():
                    self.tensor_component_configs['sub_R'] = 1
                    component['double_factor'] = False

        lambdas = self.extract_reg_terms(parameters)
        init_dict = self.construct_init_dict(parameters)
        self.train_config = self.extract_training_params(parameters)
        self.tensor_component_configs['temporal_tag'] = self.temporal_tag
        if self.bayesian:
            if self.latent_scale:
                self.model = varitional_KFT_scale(initialization_data=init_dict, shape_permutation=self.shape_permutation,
                                                  cuda=self.device, config=self.tensor_component_configs, old_setup=self.old_setup, lambdas=lambdas)
            else:
                self.model = variational_KFT(initialization_data=init_dict,shape_permutation=self.shape_permutation,
                                             cuda=self.device, config=self.tensor_component_configs, old_setup=self.old_setup, lambdas=lambdas)
        else:
            if self.latent_scale:
                self.model = KFT_scale(initialization_data=init_dict, cuda=self.device,shape_permutation=self.shape_permutation,
                                       config=self.tensor_component_configs, old_setup=self.old_setup, lambdas=lambdas)
            else:
                self.model = KFT(initialization_data=init_dict, cuda=self.device,shape_permutation=self.shape_permutation,
                                 config=self.tensor_component_configs, old_setup=self.old_setup, lambdas=lambdas)
        print(self.model)

    def start_training(self,parameters):
        print(parameters)
        self.parameters = parameters
        if self.cuda:
            self.model = self.model.to(self.device)
        self.dataloader = get_dataloader_tensor(self.data_path, seed=self.seed, mode='train',
                                                 bs_ratio=parameters['batch_size_ratio'],split_mode=self.split_mode)
        self.dataloader.chunks = self.train_config['chunks']
        torch.cuda.empty_cache()
        if self.bayesian:
            self.train(train_means=True)
            print('sigma train')
            total_cal_error_val,total_cal_error_test,val_cal_dict,test_cal_dict,val_loss_final,test_loss_final,predictions = self.train(
                train_means=False)
            self.hyperits_i+=1
            self.reset_model_dataloader()
            return total_cal_error_val,total_cal_error_test,val_cal_dict,test_cal_dict,val_loss_final,test_loss_final,predictions

        else:
            val_loss_final, test_loss_final = self.train(train_means=True)
            self.hyperits_i+=1
            self.reset_model_dataloader()
            return val_loss_final, test_loss_final

    def reset_model_dataloader(self):
        del self.model
        del self.dataloader
        torch.cuda.empty_cache()

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
            # print(Y_preds.shape)
            df = para_summary(Y_preds)
            df['y_true'] = self.dataloader.Y.numpy()
            total_cal_error,cal_dict ,predictions  =  calculate_calibration_objective(df,self.dataloader.X)
            return total_cal_error,cal_dict ,predictions
    def __call__(self, parameters):
        # for i in range(2):
        #     try: #Try two times
        torch.cuda.empty_cache()
        self.init(parameters)
        if self.bayesian:
            total_cal_error_val,total_cal_error_test,val_cal_dict,test_cal_dict,val_loss_final,test_loss_final,predictions = self.start_training(parameters)
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
            val_loss_final, test_loss_final = self.start_training(parameters)
            if not np.isinf(val_loss_final):
                torch.cuda.empty_cache()
                return {'loss': -val_loss_final, 'status': STATUS_OK, 'test_loss': -test_loss_final}
            # except Exception as e:
            #     print(e)
            #     torch.cuda.empty_cache()
        return {'loss': np.inf, 'status': STATUS_FAIL, 'test_loss': np.inf}

    def  get_kernel_vals(self,desc):
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
        return reg_params

    def construct_kernel_params(self,side_info_dims,parameters):
        kernel_param = {}
        for i in range(len(side_info_dims)):
            dim = side_info_dims[i]
            if dim in self.available_side_info_dims:
                k,nu = self.get_kernel_vals(parameters[f'kernel_{dim}_choice'])
                kernel_param[i+1] = {'ARD':parameters[f'ARD_{dim}'],'nu':nu,'kernel_type':k,'p':1.0}
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
            items['init_scale'] = self.init_max
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

    def dump_model(self, val_loss, test_loss, i, parameters):
        print(f"dumping model @ val_loss = {val_loss} test_loss = {test_loss}")
        torch.save({'model_state_dict':self.model.state_dict(),
                    'test_loss': test_loss,
                    'val_loss': val_loss,
                    'i':i,
                    'parameters':parameters
                    }, f'{self.save_path}/{self.name}_model_hyperit={i}.pt')

    def load_dumped_model(self,i):
        model_dict = torch.load(f'{self.save_path}/{self.name}_model_hyperit={i}.pt')
        self.model.load_state_dict(model_dict['model_state_dict'])

    def extract_training_params(self,parameters):
        training_params = {}
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
        training_params['batch_ratio'] = parameters['batch_size_ratio']
        training_params['chunks'] = self.chunks
        training_params['dual'] = self.dual
        if not self.task=='regression':
            training_params['pos_weight'] = self.pos_weight
        if self.bayesian:
            training_params['sigma_y'] = parameters['reg_para']

        return training_params

    def run_hyperparam_opt(self):
        self.hyperits_i = 1
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

