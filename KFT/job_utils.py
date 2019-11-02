import torch
from KFT.KFT_fp_16 import KFT, variational_KFT
from tqdm import tqdm
from torch.nn.modules.loss import _Loss
from apex import amp
import apex
import pickle
import os
from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK
from KFT.util import get_dataloader_tensor,print_model_parameters,print_ls_gradients
from sklearn import metrics

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

def update_opt_lr(opt,lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


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
    for p in range(sub_epoch):
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
        if p%train_config['train_loss_interval_print']==0:
            print_ls_gradients(model)
            print(f'reg_term it {p}: {reg.data}')
            print(f'train_loss it {p}: {pred_loss.data}')


def train(model,train_config,dataloader_train, dataloader_val, dataloader_test):
    opt = torch.optim.Adam(model.parameters(), lr=train_config['V_lr'])
    if train_config['fp_16']:
        if train_config['fused']:
            del opt
            opt = apex.optimizers.FusedAdam(model.parameters(), lr=train_config['V_lr'])
            [model], [opt] = amp.initialize([model],[opt], opt_level='O1',num_losses=1)
        else:
            [model], [opt] = amp.initialize([model],[opt], opt_level='O1',num_losses=1)
    if train_config['task']=='reg':
        loss_func = torch.nn.MSELoss()
    else:
        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=train_config['pos_weight'])

    for i in tqdm(range(train_config['epochs']+1)):
        print('V')
        model.turn_off_kernel_mode()
        update_opt_lr(opt,train_config['V_lr'])
        train_loop(model, dataloader_train, loss_func, opt, train_config,train_config['sub_epoch_V'])

        calculate_loss_no_grad(model,dataloader_val,loss_func,train_config=train_config,loss_type='val',index=i)
        print('ls')
        model.turn_on_kernel_mode()
        update_opt_lr(opt,train_config['ls_lr'])
        train_loop(model, dataloader_train, loss_func, opt, train_config,train_config['sub_epoch_ls'])

    #one last epoch for good measure
    model.turn_off_kernel_mode()
    update_opt_lr(opt, train_config['V_lr'])
    train_loop(model, dataloader_train, loss_func, opt, train_config,train_config['sub_epoch_V'])

    val_loss = calculate_loss_no_grad(model, dataloader_val, loss_func,train_config=train_config, loss_type='val', index=0)

    val_loss_final = val_loss
    test_loss_final = calculate_loss_no_grad(model,dataloader_test,loss_func,train_config=train_config,loss_type='test',index=0)

    return val_loss_final,test_loss_final,model

class job_object():
    def __init__(self,side_info_dict,tensor_architecture,other_configs,seed):
        """
        :param side_info_dict: Dict containing side info EX) {i:{'data':side_info,'temporal':True}}
        :param tensor_architecture: Tensor architecture  EX) {0:{ii:[0,1],...}
        """
        self.side_info = side_info_dict
        self.tensor_architecture = tensor_architecture
        self.hyper_parameters = {}
        self.a = other_configs['reg_para_a']
        self.b = other_configs['reg_para_b']
        self.a_ = other_configs['batch_size_a']
        self.b_ = other_configs['batch_size_b'] #1.0 max
        self.fp_16 = other_configs['fp_16']
        self.fused = other_configs['fused']
        self.hyperits = other_configs['hyperits']
        self.save_path = other_configs['save_path']
        self.name = other_configs['job_name']
        self.task = other_configs['task']
        self.epochs = other_configs['epochs']
        self.bayesian = other_configs['bayesian']
        self.data_path = other_configs['data_path']
        self.cuda = other_configs['cuda']
        self.device = other_configs['device']
        self.train_loss_interval_print  = other_configs['train_loss_interval_print']
        self.sub_epoch_V = other_configs['sub_epoch_V']
        self.sub_epoch_ls = other_configs['sub_epoch_ls']
        self.config = other_configs['config']
        self.seed = seed
        self.trials = Trials()
        self.define_hyperparameter_space()

    def define_hyperparameter_space(self):
        self.hyperparameter_space = {}
        for dim,val in self.side_info.items():
            if val['temporal']:
                self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', ['rbf', 'matern_1', 'matern_2', 'matern_3', 'periodic'])
            else:
                self.hyperparameter_space[f'kernel_{dim}_choice'] = hp.choice(f'kernel_{dim}_choice', ['rbf', 'matern_1', 'matern_2', 'matern_3'])
            self.hyperparameter_space[f'ARD_{dim}'] = hp.choice(f'ARD_{dim}', [True,False])
        self.hyperparameter_space['reg_para'] = hp.uniform('reg_para', self.a, self.b)
        self.hyperparameter_space['batch_size_ratio'] = hp.uniform('batch_size_ratio', self.a_, self.b_)
        self.hyperparameter_space['lr_1'] = hp.uniform('lr_1', 1e-3*9, 1e-2)
        self.hyperparameter_space['lr_2'] = hp.uniform('lr_2', 1e-3, 1e-2)
        #Do some sort of gradient clipping... updates might break the lenghtscale! Lengthscale is completely blown up; either clip or scale

    def __call__(self, parameters):
        #TODO do tr
        init_dict = self.construct_init_dict(parameters)
        train_config = self.extract_training_params(parameters)
        if self.bayesian:
            if self.cuda:
                model = variational_KFT(initializaiton_data_frequentist=init_dict,KL_weight=parameters['reg_para'],cuda=self.device).to(self.device)
            else:
                model = variational_KFT(initializaiton_data_frequentist=init_dict,KL_weight=parameters['reg_para'],cuda='cpu')
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
        ref_met = 'R2' if self.task=='reg' else 'auc'
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
            k,nu = self.get_kernel_vals(parameters[f'kernel_{dim}_choice'])
            kernel_param[i+1] = {'ARD':parameters[f'ARD_{dim}'],'ls_factor':1.0,'nu':nu,'kernel_type':k,'p':1.0}
        return kernel_param

    def construct_side_info_params(self,side_info_dims):
        side_params = {}
        for i in range(len(side_info_dims)):
            dim = side_info_dims[i]
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
        return init_dict

    def extract_training_params(self,parameters):
        training_params = {}
        training_params['fp_16'] = self.fp_16
        training_params['fused'] = self.fused
        training_params['task'] = self.task
        training_params['epochs'] = self.epochs
        training_params['V_lr'] = parameters['lr_1']
        training_params['ls_lr'] = parameters['lr_2']
        training_params['device'] = self.device
        training_params['cuda'] = self.cuda
        training_params['train_loss_interval_print']=self.train_loss_interval_print
        training_params['sub_epoch_V']=self.sub_epoch_V
        training_params['sub_epoch_ls']=self.sub_epoch_ls
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
                    open(self.save_path + self.name + '.p',
                         "wb"))

