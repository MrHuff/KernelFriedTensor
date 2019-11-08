import torch
from KFT.KFT_fp_16 import KFT, variational_KFT
from tqdm import tqdm
from torch.nn.modules.loss import _Loss
from apex import amp
import apex
import pickle
import os
import warnings
from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK
from KFT.util import get_dataloader_tensor,print_model_parameters,get_free_gpu,process_old_setup,concat_old_side_info,load_side_info,print_ls_gradients
from sklearn import metrics
from KFT.lookahead_opt import Lookahead
import numpy as np

def run_job_func(args):
    print(args)
    with warnings.catch_warnings():  # There are some autograd issues fyi, might wanna fix it sooner or later
        warnings.simplefilter("ignore")
        gpu = get_free_gpu(8)
        PATH = args['PATH']
        if not os.path.exists(PATH + 'all_data.pt'):
            process_old_setup(PATH, tensor_name=args['tensor_name'])
            concat_old_side_info(PATH, args['side_info_name'])

        side_info = load_side_info(side_info_path=PATH, indices=args['side_info_order'])
        shape = pickle.load(open(PATH + 'full_tensor_shape.pickle', 'rb'))
        for i in args['temporal_tag']:
            side_info[i]['temporal'] = True
        print(f'USING GPU:{gpu[0]}')
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
            'device': f'cuda:{gpu[0]}',
            'train_loss_interval_print': args['sub_epoch_V'] // 10,
            'sub_epoch_V': args['sub_epoch_V'],
            'sub_epoch_ls': args['sub_epoch_ls'],
            'sub_epoch_prime': args['sub_epoch_prime'],
            'config': {'full_grad': args['full_grad']},
            'shape':shape,
            'architecture': args['architecture'],
            'max_R': args['max_R']
        }
        j = job_object(
            side_info_dict=side_info,
            configs=other_configs,
            seed=args['seed']
        )
        j.run_hyperparam_opt()


def get_tensor_architectures(i,shape,R=2):
    TENSOR_ARCHITECTURES = {
        0:{
           0:{'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':R,'has_side_info':True},
           1: {'ii': [1], 'r_1': R, 'n_list': [shape[1]], 'r_2': R, 'has_side_info': True}, #Magnitude of kernel sum
           2: {'ii': [2], 'r_1': R, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True},
           },
        1:{
           0:{'ii':[0],'r_1':1,'n_list':[shape[0]],'r_2':R,'has_side_info':True},
           1: {'ii': [1,2], 'r_1': R, 'n_list': [shape[1],shape[2]], 'r_2': 1, 'has_side_info': True}, #Magnitude of kernel sum
           },
        2: {
            0: {'ii': [0,1], 'r_1': 1, 'n_list': [shape[0],shape[1]], 'r_2': R, 'has_side_info': True},
            1: {'ii': [2], 'r_1': R, 'n_list': [shape[2]], 'r_2': 1, 'has_side_info': True},
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

def auc_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(Y.cpu().numpy(), y_pred, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        return auc

def calculate_loss_no_grad(model,dataloader,loss_func,train_config,loss_type='type',index=0,task='reg'):
    loss_list = []
    y_s = []
    _y_preds = []
    with torch.no_grad():
        X, y = dataloader.get_batch()
        if train_config['cuda']:
            X = X.to(train_config['device'])
            y = y.to(train_config['device'])
        if train_config['bayesian']:
            y_pred = model.mean_forward(X)
        else:
            y_pred, _ = model(X)
        loss = loss_func(y_pred, y)
        loss_list.append(loss)
        y_s.append(y)
        _y_preds.append(y_pred)

    total_loss = torch.tensor(loss_list).mean().data
    Y = torch.cat(y_s)
    y_preds = torch.cat(_y_preds)
    if task=='reg':
        var_Y = Y.var()
        ref_metric = 1.-total_loss/var_Y
    else:
        ref_metric = auc_check(y_preds,Y)
    print(f'{loss_type} ref metric epoch {index}: {ref_metric}')
    return ref_metric

def train_loop(model, dataloader, loss_func, opt, train_config,sub_epoch):
    ERROR = False
    train_config['reset'] = 1e-2
    for p in range(sub_epoch+1):
        X,y = dataloader.get_batch()
        if train_config['cuda']:
            X = X.to(train_config['device'])
            y = y.to(train_config['device'])
        y_pred, reg = model(X)
        pred_loss = loss_func(y_pred, y)
        total_loss = pred_loss + reg
        opt.zero_grad()
        if train_config['fp_16']:
            with amp.scale_loss(total_loss, opt, loss_id=0) as loss_scaled:
                loss_scaled.backward()
        else:
            total_loss.backward()
        opt.step()
        with torch.no_grad():
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any() or torch.isnan(reg) or torch.isinf(reg):
                ERROR = True
            if p%train_config['train_loss_interval_print']==0:
                if train_config['bayesian']:
                    y_pred= model.mean_forward(X)
                    mean_pred_loss = loss_func(y_pred, y)
                    if (y_pred==0).all() or y_pred.sum()==0:
                        print('dead model_reinit')
                        for p in model.parameters():
                            reinit_model(p,train_config['reset'])
                        train_config['reset'] = train_config['reset']*2

                    print(f'reg_term it {p}: {reg.data}')
                    print(f'train_loss it {p}: {pred_loss.data}')
                    print(f'mean_loss it {p}: {mean_pred_loss.data}')
                else:
                    if (y_pred == 0).all() or y_pred.sum() == 0:
                        print('dead model_reinit')
                        for p in model.parameters():
                            reinit_model(p, train_config['reset'])
                        train_config['reset'] = train_config['reset']*2
                    print(f'reg_term it {p}: {reg.data}')
                    print(f'train_loss it {p}: {pred_loss.data}')
        if ERROR:
            return ERROR
    return ERROR

def reinit_model(para,scale):
    torch.nn.init.uniform_(para,0,scale*2)

def opt_reinit(train_config,model,lr_param):
    model = model.float()
    opt = torch.optim.Adam(model.parameters(), lr=train_config[lr_param], amsgrad=False)
    opt = Lookahead(opt)
    if train_config['fp_16']:
        if train_config['fused']:
            del opt
            opt = apex.optimizers.FusedAdam(model.parameters(), lr=train_config[lr_param])
            opt = Lookahead(opt)
            [model], [opt] = amp.initialize([model],[opt], opt_level='O1',num_losses=1,)
        else:
            [model], [opt] = amp.initialize([model],[opt], opt_level='O1',num_losses=1)
    return model,opt

def train(model,train_config,dataloader_train, dataloader_val, dataloader_test):

    if train_config['task']=='reg':
        loss_func = torch.nn.MSELoss()
    else:
        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=train_config['pos_weight'])

    """
    Warm up
    """
    print('V')  # Regular ADAM does the job
    model.turn_on_V()
    model, opt = opt_reinit(train_config, model, 'V_lr')
    ERROR = train_loop(model, dataloader_train, loss_func, opt, train_config, train_config['sub_epoch_V'])
    if ERROR:
        return np.inf,np.inf,np.inf
    print('prime')
    model.turn_on_prime()
    model, opt = opt_reinit(train_config, model, 'prime_lr')
    ERROR = train_loop(model, dataloader_train, loss_func, opt, train_config, train_config['sub_epoch_prime'])
    if ERROR:
        return np.inf, np.inf, np.inf

    for i in tqdm(range(train_config['epochs']+1)):

        print('ls')
        model.turn_on_kernel_mode()
        model,opt = opt_reinit(train_config,model,'ls_lr')
        ERROR = train_loop(model, dataloader_train, loss_func, opt, train_config,train_config['sub_epoch_ls'])
        if ERROR:
            return np.inf, np.inf, np.inf

        print('V') #Regular ADAM does the job
        model.turn_on_V()
        model,opt = opt_reinit(train_config,model,'V_lr')
        ERROR = train_loop(model, dataloader_train, loss_func, opt, train_config,train_config['sub_epoch_V'])
        if ERROR:
            return np.inf, np.inf, np.inf

        calculate_loss_no_grad(model,dataloader_val,loss_func,train_config=train_config,loss_type='val',index=i)

        print('prime')
        model.turn_on_prime()
        model,opt = opt_reinit(train_config,model,'prime_lr')
        ERROR = train_loop(model, dataloader_train, loss_func, opt, train_config,train_config['sub_epoch_prime'])
        if ERROR:
            return np.inf, np.inf, np.inf

    val_loss = calculate_loss_no_grad(model, dataloader_val, loss_func,train_config=train_config, loss_type='val', index=0)
    val_loss_final = val_loss
    test_loss_final = calculate_loss_no_grad(model,dataloader_test,loss_func,train_config=train_config,loss_type='test',index=0)

    return val_loss_final,test_loss_final,model

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
        self.name = f'bayesian_{seed}' if configs['bayesian'] else f'frequentist_{seed}'
        self.task = configs['task']
        self.epochs = configs['epochs']
        self.bayesian = configs['bayesian']
        self.data_path = configs['data_path']
        self.cuda = configs['cuda']
        self.device = configs['device']
        self.train_loss_interval_print  = configs['train_loss_interval_print']
        self.sub_epoch_V = configs['sub_epoch_V']
        self.sub_epoch_prime = configs['sub_epoch_prime']
        self.sub_epoch_ls = configs['sub_epoch_ls']
        self.config = configs['config']
        self.architecture = configs['architecture']
        self.shape = configs['shape']
        self.max_R = configs['max_R']
        self.seed = seed
        self.trials = Trials()
        self.define_hyperparameter_space()

    def define_hyperparameter_space(self):
        self.hyperparameter_space = {}
        self.available_side_info_dims = []
        t_act = get_tensor_architectures(self.architecture,self.shape,2)
        for dim,val in self.side_info.items():
            if self.fp_16:
                self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', ['rbf','periodic'])
            else:
                self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', ['matern_1', 'matern_2', 'matern_3', 'periodic','rbf'])
            self.hyperparameter_space[f'ARD_{dim}'] = hp.choice(f'ARD_{dim}', [True,False])
            self.available_side_info_dims.append(dim)
        self.hyperparameter_space['reg_para'] = hp.uniform('reg_para', self.a, self.b)
        self.hyperparameter_space['batch_size_ratio'] = hp.uniform('batch_size_ratio', self.a_, self.b_)
        self.hyperparameter_space['R'] = hp.choice('R', np.arange(2,self.max_R+1,dtype=int))
        self.hyperparameter_space['lr_1'] = hp.choice('lr_1', [1e-3,1e-2]) #Very important for convergence
        self.hyperparameter_space['lr_2'] = hp.choice('lr_2', [1e-3,1e-2,1e-1] if not self.bayesian else [1e-4,1e-3]) #Very important for convergence
        self.hyperparameter_space['lr_3'] = hp.choice('lr_3', [1e-3,1e-2,1e-1] if not self.bayesian else [1e-3, 1e-2]) #Very important for convergence
        for i in t_act.keys():
            self.hyperparameter_space[f'init_scale_{i}'] = hp.choice(f'init_scale_{i}',[1e-3,1e-2,1e-1])
            if self.bayesian:
                self.hyperparameter_space[f'multivariate_{i}'] = hp.choice(f'multivariate_{i}',[True,False])

    def __call__(self, parameters):
        #TODO do tr
        # try:
        self.tensor_architecture = get_tensor_architectures(self.architecture,self.shape,parameters['R'])
        init_dict = self.construct_init_dict(parameters)
        train_config = self.extract_training_params(parameters)
        print(parameters)
        if self.bayesian:
            if self.cuda:
                model = variational_KFT(initializaiton_data_frequentist=init_dict,KL_weight=parameters['reg_para'],cuda=self.device,config=self.config).to(self.device)
            else:
                model = variational_KFT(initializaiton_data_frequentist=init_dict,KL_weight=parameters['reg_para'],cuda='cpu',config=self.config)
        else:
            if self.cuda:
                model = KFT(initializaiton_data=init_dict,lambda_reg=parameters['reg_para'],cuda=self.device,config=self.config).to(self.device)
            else:
                model = KFT(initializaiton_data=init_dict,lambda_reg=parameters['reg_para'],cuda='cpu',config=self.config)
        print_model_parameters(model)
        dataloader_train = get_dataloader_tensor(self.data_path,seed = self.seed,mode='train',bs_ratio=parameters['batch_size_ratio'])
        dataloader_val = get_dataloader_tensor(self.data_path,seed = self.seed,mode='val',bs_ratio=parameters['batch_size_ratio'])
        dataloader_test = get_dataloader_tensor(self.data_path,seed = self.seed,mode='test',bs_ratio=parameters['batch_size_ratio'])
        val_loss_final,test_loss_final,model = train(model=model,train_config=train_config,dataloader_train=dataloader_train,dataloader_val=dataloader_val,dataloader_test=dataloader_test)
        # except Exception as e:
        #     print(e)
        #     val_loss_final = np.inf
        #     test_loss_final = np.inf
        ref_met = 'R2' if self.task == 'reg' else 'auc'
        return {'loss': val_loss_final, 'status': STATUS_OK, f'test_{ref_met}': test_loss_final}

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
            kernel_param  = self.construct_kernel_params(side_info_dims,parameters)
            side_param = self.construct_side_info_params(side_info_dims)
            component_init['kernel_para'] = kernel_param
            component_init['side_info'] = side_param
            component_init['init_scale'] = parameters[f'init_scale_{key}']
            if self.bayesian:
                component_init['multivariate'] = parameters[f'multivariate_{key}']
        return init_dict

    def extract_training_params(self,parameters):
        training_params = {}
        training_params['fp_16'] = self.fp_16
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
        training_params['sub_epoch_prime']=self.sub_epoch_prime
        training_params['sub_epoch_ls']=self.sub_epoch_ls
        training_params['bayesian'] = self.bayesian
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

