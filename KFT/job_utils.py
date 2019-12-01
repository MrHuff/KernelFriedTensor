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
# from KFT.lookahead_opt import Lookahead
import numpy as np
import timeit
def run_job_func(args):
    print(args)
    with warnings.catch_warnings():  # There are some autograd issues fyi, might wanna fix it sooner or later
        warnings.simplefilter("ignore")
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
        primal_dims = list(shape)
        for key,val in side_info.items():
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
            'config': {'full_grad': args['full_grad'],'deep_kernel':args['deep_kernel']},
            'shape':shape,
            'architecture': args['architecture'],
            'max_R': args['max_R'],
            'max_lr':args['max_lr'],
            'old_setup':args['old_setup'],
            'latent_scale':args['latent_scale'],
            'chunks':args['chunks'],
            'primal_list': primal_dims,
            'dual':args['dual']

        }
        j = job_object(
            side_info_dict=side_info,
            configs=other_configs,
            seed=args['seed']
        )
        j.run_hyperparam_opt()

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
           0:{'primal_list':[primal_dims[0]],'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':R,'has_side_info':True,'r_1_latent':1,'r_2_latent':R_scale},
           1:{'primal_list':[primal_dims[1]],'ii': [1], 'r_1': R, 'n_list': [shape[1]], 'r_2': R, 'has_side_info': True,'r_1_latent':R_scale,'r_2_latent':R_scale}, #Magnitude of kernel sum
           2:{'primal_list':[primal_dims[2]],'ii': [2], 'r_1': R, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True,'r_1_latent':R_scale,'r_2_latent':1},
           },
        1:{
           0: {'primal_list':[primal_dims[0]],'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':R,'has_side_info':True,'r_1_latent':1,'r_2_latent':R_scale},
           1: {'primal_list':[primal_dims[1],primal_dims[2]],'ii': [1,2], 'r_1': R, 'n_list': [shape[1],shape[2]], 'r_2': 1, 'has_side_info': True,'r_1_latent':R_scale,'r_2_latent':1}, #Magnitude of kernel sum
           },
        2: {
            0: {'primal_list':[primal_dims[0]],'ii': [0,1], 'r_1': 1, 'n_list': [shape[0],shape[1]], 'r_2': R, 'has_side_info': True,'r_1_latent':1,'r_2_latent':R_scale},
            1: {'primal_list':[primal_dims[1],primal_dims[2]],'ii': [2], 'r_1': R, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True,'r_1_latent':R_scale,'r_2_latent':1},
        },
        3: {
            0: {'primal_list':[primal_dims[0]],'ii': [0], 'r_1': 1, 'n_list': [shape[0]], 'r_2': R, 'has_side_info': False,'r_1_latent':1,'r_2_latent':R_scale},
            1: {'primal_list':[primal_dims[1],primal_dims[2]],'ii': [1, 2], 'r_1': R, 'n_list': [shape[1], shape[2]], 'r_2': 1, 'has_side_info': False,'r_1_latent':R_scale,'r_2_latent':1},
        },
        4: {
            0: {'primal_list':[primal_dims[0]],'ii': [0], 'r_1': 1, 'n_list': [shape[0]], 'r_2': R, 'has_side_info': False,'r_1_latent':1,'r_2_latent':R_scale},
            1: {'primal_list':[primal_dims[1],primal_dims[2]],'ii': [1, 2], 'r_1': R, 'n_list': [shape[1], shape[2]], 'r_2':  1, 'has_side_info': True,'r_1_latent':R_scale,'r_2_latent':1},
        },
        5: {
            0: {'primal_list':[primal_dims[0]],'ii': [0], 'r_1': 1, 'n_list': [shape[0]], 'r_2': R, 'has_side_info': False,'r_1_latent':1,'r_2_latent':R_scale},
            1: {'primal_list':[primal_dims[1]],'ii': [1], 'r_1': R, 'n_list': [shape[1]], 'r_2': R, 'has_side_info': False,'r_1_latent':R_scale,'r_2_latent':R_scale},  # Magnitude of kernel sum
            2: {'primal_list':[primal_dims[2]],'ii': [2], 'r_1': R, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': False,'r_1_latent':R_scale,'r_2_latent':1},
        },
        6: {
            0: {'primal_list':[primal_dims[0]],'ii': [0], 'r_1': 1, 'n_list': [shape[0]], 'r_2': R, 'has_side_info': False,'r_1_latent':1,'r_2_latent':R_scale},
            1: {'primal_list':[primal_dims[1]],'ii': [1], 'r_1': R, 'n_list': [shape[1]], 'r_2': R, 'has_side_info': False,'r_1_latent':R_scale,'r_2_latent':R_scale},  # Magnitude of kernel sum
            2: {'primal_list':[primal_dims[2]],'ii': [2], 'r_1': R, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True,'r_1_latent':R_scale,'r_2_latent':1},
        },
        7: {
            0: {'primal_list':[primal_dims[0]],'ii': [0], 'r_1': 1, 'n_list': [shape[0]], 'r_2': R, 'has_side_info': True, 'r_1_latent': 1,'r_2_latent': R_scale},
            1: {'primal_list':[primal_dims[1]],'ii': [1], 'r_1': R, 'n_list': [shape[1]], 'r_2': R, 'has_side_info': False, 'r_1_latent': R_scale,'r_2_latent': R_scale},  # Magnitude of kernel sum
            2: {'primal_list':[primal_dims[2]],'ii': [2], 'r_1': R, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True, 'r_1_latent': R_scale,
                'r_2_latent': 1},
        },
        8: {
            0: {'primal_list':[primal_dims[0]],'ii': [0], 'r_1': 1, 'n_list': [shape[0]], 'r_2': R, 'has_side_info': False, 'r_1_latent': 1,'r_2_latent': R_scale},
            1: {'primal_list':[primal_dims[1]],'ii': [1], 'r_1': R, 'n_list': [shape[1]], 'r_2': R, 'has_side_info': True, 'r_1_latent': R_scale,'r_2_latent': R_scale},  # Magnitude of kernel sum
            2: {'primal_list':[primal_dims[2]],'ii': [2], 'r_1': R, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True, 'r_1_latent': R_scale,
                'r_2_latent': 1},
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
    return torch.mean(y**2 -2*y*middle_term+last_term)

def auc_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(Y.cpu().numpy(), y_pred, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        return auc

def calculate_loss_no_grad(model,dataloader,train_config,task='reg',mode='val'):
    loss_list = []
    y_s = []
    _y_preds = []
    dataloader.set_mode(mode)
    with torch.no_grad():
        for i in range(dataloader.chunks):
            X, y = dataloader.get_chunk(i)
            if train_config['cuda']:
                X = X.to(train_config['device'])
                y = y.to(train_config['device'])
            loss,y_pred = correct_validation_loss(X, y, model, train_config)
            loss_list.append(loss)
            y_s.append(y)
            _y_preds.append(y_pred)
        total_loss = torch.tensor(loss_list).mean().data
        Y = torch.cat(y_s,dim=0)
        y_preds = torch.cat(_y_preds)
        if task=='reg':
            var_Y = Y.var()
            ref_metric = 1.-total_loss/var_Y
            ref_metric = ref_metric.numpy()
        else:
            ref_metric = auc_check(y_preds,Y)
    return ref_metric

def train_monitor(total_loss,reg,pred_loss,model,y_pred,train_config,p):
    with torch.no_grad():
        ERROR = False
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print('FOUND INF/NAN RIP, RESTARTING')
            print(reg)
            print(pred_loss)
            print(y_pred.mean())
            ERROR = True
        if (y_pred == 0).all():
            fac = train_config['reset']
            print(f'dead model_reinit factor: {fac}')
            for n, param in model.named_parameters():
                if 'core_param' in n:
                    param.normal_(0, train_config['reset'])
            train_config['reset'] = train_config['reset'] * 1.1

        if p % train_config['train_loss_interval_print'] == 0:
            print(f'reg_term it {p}: {reg.data}')
            print(f'train_loss it {p}: {pred_loss.data}')
    return ERROR

def correct_validation_loss(X,y,model,train_config):
    if train_config['task'] == 'reg':
        loss_func = torch.nn.MSELoss()
        if train_config['bayesian']:
            y_pred, _,_ = model(X)
        else:
            y_pred, _ = model(X)
        pred_loss = loss_func(y_pred,y.squeeze())
    else:
        loss_func = torch.nn.BCELoss()
        y_pred, reg = model(X)
        pred_loss = loss_func(y_pred, y.squeeze())
    return pred_loss,y_pred

def correct_forward_loss(X,y,model,train_config,loss_func):
    if train_config['task']=='reg' and train_config['bayesian']:
            y_pred,last_term, reg = model(X)
            pred_loss = loss_func(y.squeeze(),y_pred,last_term)
    else:
        y_pred, reg = model(X)
        pred_loss = loss_func(y_pred, y.squeeze())
    return pred_loss+reg,reg,pred_loss,y_pred

def train_loop(model,opt, dataloader, loss_func, train_config,sub_epoch,warmup=False):

    lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=train_config['patience'],factor=0.5)
    print_garbage()
    dataloader.rerandomize()
    n = dataloader.train_chunks
    torch.cuda.empty_cache()
    for p in range(n):
        dataloader.set_mode('train')
        X,y = dataloader.get_chunk(p)
        if train_config['cuda']:
            X = X.to(train_config['device'])
            y = y.to(train_config['device'])
        total_loss,reg,pred_loss,y_pred = correct_forward_loss(X,y,model,train_config,loss_func)
        opt.zero_grad()
        if train_config['fp_16'] and not warmup:
            with train_config['amp'].scale_loss(total_loss, opt, loss_id=0) as loss_scaled:
                loss_scaled.backward()
        else:
            total_loss.backward()
        opt.step()
        lrs.step(total_loss)
        ERROR = train_monitor(total_loss,reg,pred_loss,model,y_pred,train_config,p)
        if ERROR:
            return ERROR
    torch.cuda.empty_cache()
    l_val = calculate_loss_no_grad(model,dataloader=dataloader,train_config=train_config,mode='val')
    torch.cuda.empty_cache()
    l_test = calculate_loss_no_grad(model,dataloader=dataloader,train_config=train_config,mode='test')
    torch.cuda.empty_cache()
    # print(f'train_error= {l_train}')
    print(f'val_error= {l_val}')
    print(f'test_error= {l_test}')


    del lrs
    return False

def print_garbage():
    obj_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # print(type(obj), obj.size())
                obj_list.append(obj)
        except:
            pass
    print(len(obj_list))

def opt_reinit(train_config,model,lr_params,warmup=False):
    if train_config['fp_16'] and not warmup:
        import apex
        from apex import amp
        amp.register_float_function(torch, 'bmm')#TODO: HALLELUJA
        if train_config['fused']:
            tmp_opt_list = []
            for lr in lr_params:
                tmp_opt_list.append(apex.optimizers.FusedAdam(model.parameters(), lr=train_config[lr])) #Calling this again makes the model completely in FP16 wtf
            [model], tmp_opt_list = amp.initialize([model],tmp_opt_list, opt_level='O1',num_losses=1)
            opts  = {x:y for x,y in zip(lr_params,tmp_opt_list)}
            model.amp = amp
            model.amp_patch(amp)
        else:
            tmp_opt_list = []
            for lr in lr_params:
                tmp_opt_list.append(torch.optim.Adam(model.parameters(), lr=train_config[lr], amsgrad=False))
            [model], tmp_opt_list = amp.initialize([model],tmp_opt_list, opt_level='O1',num_losses=1)
            opts  = {x:y for x,y in zip(lr_params,tmp_opt_list)}
            [model], tmp_opt_list = amp.initialize([model],tmp_opt_list, opt_level='O1',num_losses=1)
            model.amp = amp
            model.amp_patch(amp)
        train_config['amp'] = amp
    else:
        tmp_opt_list = []
        for lr in lr_params:
            tmp_opt_list.append(torch.optim.Adam(model.parameters(), lr=train_config[lr], amsgrad=False))
        opts = {x: y for x, y in zip(lr_params, tmp_opt_list)}
    return model,opts

def opt_32_reinit(model,train_config,lr):
    opt = torch.optim.Adam(model.parameters(), lr=train_config[lr], amsgrad=False)
    return opt

def setup_runs(model,train_config,warmup):
    loss_func = get_loss_func(train_config)
    ERROR = False
    kernel, deep_kernel = model.has_kernel_component()
    train_dict = {0: {'para': 'V_lr', 'call': model.turn_on_V}}
    train_list = [0]
    if not train_config['old_setup']:
        train_dict[1] = {'para': 'prime_lr', 'call': model.turn_on_prime}
        train_list.append(1)
    if train_config['dual']:
        if kernel:
            train_dict[2] = {'para': 'ls_lr', 'call': model.turn_on_kernel_mode}
            train_list.insert(-1, 2)
        if deep_kernel:
            train_dict[3] = {'para': 'deep_lr', 'call': model.turn_on_deep_kernel}
            train_list.insert(-2, 3)
    if warmup:
        train_list = [0]
        if not train_config['old_setup']:
            train_list.append(1)
    lrs = [v['para'] for v in train_dict.values()]
    model, opts = opt_reinit(train_config, model, lrs, warmup=warmup)
    return model,opts,loss_func,ERROR,train_list,train_dict

def outer_train_loop(model,opts,loss_func,ERROR,train_list,train_dict, train_config, dataloader, warmup=False):
    for i in train_list:
        print_garbage()
        settings = train_dict[i]
        f = settings['call']
        lr = settings['para']
        f()
        print(lr)
        opt = opts[lr]
        ERROR = train_loop(model,opt, dataloader, loss_func, train_config, train_config['sub_epoch_V'], warmup=warmup)
        print_garbage()
        if ERROR:
            return ERROR
        print(torch.cuda.memory_cached() / 1e6)
        print(torch.cuda.memory_allocated() / 1e6)
        torch.cuda.empty_cache()
    return ERROR

def train(model, train_config, dataloader):
    train_config['reset'] = 1.0
    model,opts,loss_func,ERROR,train_list,train_dict = setup_runs(model,train_config,warmup=False)
    for i in range(train_config['epochs']+1):
        ERROR = outer_train_loop(model,opts,loss_func,ERROR,train_list,train_dict, train_config, dataloader, warmup=False)
        if ERROR:
            return -np.inf, -np.inf
    val_loss_final = calculate_loss_no_grad(model,dataloader=dataloader, train_config=train_config,mode='val')
    test_loss_final = calculate_loss_no_grad(model,dataloader=dataloader,train_config=train_config,mode='test')
    del model
    del opts
    print(val_loss_final,test_loss_final)
    return val_loss_final,test_loss_final

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
        self.device = configs['device']
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
        self.seed = seed
        self.trials = Trials()
        self.define_hyperparameter_space()

    def define_hyperparameter_space(self):
        self.hyperparameter_space = {}
        self.available_side_info_dims = []
        t_act = get_tensor_architectures(self.architecture,self.shape,self.primal_dims, 2)
        for dim,val in self.side_info.items():
            self.available_side_info_dims.append(dim)
            if self.dual:
                if self.fp_16:
                    if val['temporal']:
                        self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', ['rbf','periodic'])
                    else:
                        self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', ['rbf'])
                else:
                    if val['temporal']:
                        self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', ['matern_1', 'matern_2', 'matern_3', 'periodic','rbf'])
                    else:
                        self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', ['matern_1', 'matern_2', 'matern_3', 'rbf'])
                self.hyperparameter_space[f'ARD_{dim}'] = hp.choice(f'ARD_{dim}', [True,False])

        self.hyperparameter_space['init_scale'] = hp.choice('init_scale', [1e-1])
        self.hyperparameter_space['reg_para'] = hp.uniform('reg_para', self.a, self.b)
        self.hyperparameter_space['batch_size_ratio'] = hp.uniform('batch_size_ratio', self.a_, self.b_)
        if self.latent_scale:
            self.hyperparameter_space['R_scale'] = hp.choice('R_scale', np.arange(1,self.max_R//2,dtype=int))
        self.hyperparameter_space['R'] = hp.choice('R', np.arange(self.max_R,self.max_R+1,dtype=int))
        self.hyperparameter_space['lr_1'] = hp.choice('lr_1', np.divide(self.lrs, 10.)) #Very important for convergence
        self.hyperparameter_space['lr_2'] = hp.choice('lr_2', self.lrs ) #Very important for convergence
        self.hyperparameter_space['lr_3'] = hp.choice('lr_3', self.lrs ) #Very important for convergence
        if self.bayesian:
            for i in t_act.keys():
                self.hyperparameter_space[f'multivariate_{i}'] = hp.choice(f'multivariate_{i}',[True,False])

    def init_and_train(self,parameters):
        self.config['dual'] = self.dual
        self.tensor_architecture = get_tensor_architectures(self.architecture, self.shape,self.primal_dims, parameters['R'],parameters['R_scale'] if self.latent_scale else 1)
        init_dict = self.construct_init_dict(parameters)
        train_config = self.extract_training_params(parameters)
        print(parameters)
        if self.bayesian:
            if self.latent_scale:
                if self.cuda:
                    model = varitional_KFT_scale(initialization_data=init_dict, KL_weight=parameters['reg_para'],
                                            cuda=self.device, config=self.config, old_setup=self.old_setup).to(
                        self.device)
                else:
                    model = varitional_KFT_scale(initialization_data=init_dict, KL_weight=parameters['reg_para'], cuda='cpu',
                                            config=self.config, old_setup=self.old_setup)
            else:
                if self.cuda:
                    model = variational_KFT(initialization_data=init_dict, KL_weight=parameters['reg_para'],
                                            cuda=self.device, config=self.config, old_setup=self.old_setup).to(self.device)
                else:
                    model = variational_KFT(initialization_data=init_dict, KL_weight=parameters['reg_para'], cuda='cpu',
                                            config=self.config, old_setup=self.old_setup)
        else:
            if self.latent_scale:
                if self.cuda:
                    model = KFT_scale(initialization_data=init_dict, lambda_reg=parameters['reg_para'], cuda=self.device,
                                config=self.config, old_setup=self.old_setup).to(self.device)
                else:
                    model = KFT_scale(initialization_data=init_dict, lambda_reg=parameters['reg_para'], cuda='cpu',
                                config=self.config, old_setup=self.old_setup)
            else:
                if self.cuda:
                    model = KFT(initialization_data=init_dict, lambda_reg=parameters['reg_para'], cuda=self.device,
                                config=self.config, old_setup=self.old_setup).to(self.device)
                else:
                    model = KFT(initialization_data=init_dict, lambda_reg=parameters['reg_para'], cuda='cpu',
                                config=self.config, old_setup=self.old_setup)
        print(model)
        dataloader = get_dataloader_tensor(self.data_path, seed=self.seed, mode='train',
                                                 bs_ratio=parameters['batch_size_ratio'])
        dataloader.chunks = train_config['chunks']
        val_loss_final, test_loss_final = train(model=model, train_config=train_config,
                                                dataloader=dataloader)
        return val_loss_final, test_loss_final

    def __call__(self, parameters):
        # try:
        for i in range(10):
            val_loss_final, test_loss_final = self.init_and_train(parameters)
            if not np.isinf(val_loss_final):
                ref_met = 'R2' if self.task == 'reg' else 'auc'
                return {'loss': -val_loss_final, 'status': STATUS_OK, f'test_{ref_met}': -test_loss_final}
        # except Exception as e:
        #     print(e)
        ref_met = 'R2' if self.task == 'reg' else 'auc'
        return {'loss': np.inf, 'status': STATUS_FAIL, f'test_{ref_met}': np.inf}

    def get_kernel_vals(self,desc):
        if 'matern_1'== desc:
            return 'matern',0.5
        elif 'matern_2' == desc:
            return 'matern',1.5
        elif 'matern_3' == desc:
            return 'matern',2.5
        else:
            return desc,None

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
            component_init = init_dict[key]
            side_info_dims = component_init['ii']
            if self.dual:
                kernel_param  = self.construct_kernel_params(side_info_dims,parameters)
            else:
                kernel_param = {}
            side_param = self.construct_side_info_params(side_info_dims)
            component_init['kernel_para'] = kernel_param
            component_init['side_info'] = side_param
            component_init['init_scale'] = parameters['init_scale']
            if self.bayesian:
                component_init['multivariate'] = parameters[f'multivariate_{key}']
        return init_dict

    def extract_training_params(self,parameters):
        training_params = {}
        training_params['fp_16'] = (self.fp_16 and not self.bayesian)
        training_params['fused'] = self.fused
        training_params['task'] = self.task
        training_params['epochs'] = self.epochs
        training_params['prime_lr'] = parameters['lr_3']
        training_params['V_lr'] = parameters['lr_2']
        training_params['ls_lr'] = parameters['lr_1']
        training_params['device'] = self.device
        training_params['cuda'] = self.cuda
        training_params['train_loss_interval_print']=self.train_loss_interval_print
        training_params['sub_epoch_V']=self.sub_epoch_V
        training_params['bayesian'] = self.bayesian
        training_params['old_setup'] = self.old_setup
        training_params['architecture'] = self.architecture
        training_params['deep_kernel'] = self.deep_kernel
        training_params['batch_ratio'] = parameters['batch_size_ratio']
        training_params['patience'] = 50
        training_params['chunks'] = self.chunks
        training_params['dual'] = self.dual
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

