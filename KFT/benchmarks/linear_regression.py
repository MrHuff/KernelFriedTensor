from hyperopt import hp,tpe,fmin,Trials,STATUS_OK,space_eval
import hyperopt
from KFT.util import get_free_gpu
import os
import time
import pickle
from KFT.benchmarks.utils import read_benchmark_data,get_auc,get_R_square
import torch
from KFT.job_utils import get_loss_func,calculate_loss_no_grad
import numpy as np

class linear_regression(torch.nn.Module):
    def __init__(self,nr_col):
        super(linear_regression, self).__init__()
        self.w = torch.nn.Parameter(torch.randn((nr_col,1)),requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn((1,1)),requires_grad=True)

    def forward(self, X):
        f = X@self.w+self.b
        return f.squeeze(), torch.mean(self.w**2)

class bayesian_linear_regression(linear_regression):
    def __init__(self, nr_col):
        super(bayesian_linear_regression, self).__init__(nr_col)
        self.w_sigma = torch.nn.Parameter(torch.randn((nr_col,1)),requires_grad=True)
        self.b_sigma = torch.nn.Parameter(torch.randn((1,1)),requires_grad=True)

    def KL(self,mean,sig):
        return torch.mean(0.5*(sig.exp()+mean**2-sig-1))

    def forward(self,X):
        a = X@self.w
        middle_term = a +self.b
        last_term = a**2 + 2*self.b*a+X**2@self.w_sigma.exp()+self.b_sigma.exp()+self.b**2
        KL_tot = self.KL(self.w,self.w_sigma)+self.KL(self.b,self.b_sigma)
        return middle_term,last_term,KL_tot

class linear_job_class():
    def __init__(self,seed,y_name,data_path,save_path,params):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.data_obj = read_benchmark_data(y_name=y_name, path=data_path, seed=seed)
        self.data_obj.FFM =False
        self.nr_cols = len(self.data_obj.n_list)
        self.hyperits = params['hyperopts']
        self.task = 'reg' if params['regression'] else 'binary'
        self.chunks= params['chunk']
        self.its = params['its']
        self.bayesian = params['bayesian']
        self.name = f'linear_{seed}'
        self.cuda=params['cuda']
        if self.cuda:
            gpu = get_free_gpu(10)[0]
            self.device = f'cuda:{gpu}'
        else:
            self.device = f'cpu'
        self.space = {
            'lambda': hp.uniform('lambda', 0.0 if not self.bayesian else 1.0, 1.0 if not self.bayesian else 2.0),
            'lr': hp.uniform('lr', 0.05, 1.0),
            'ratio': hp.uniform('ratio', 1e-6, 0.1),
        }

    def get_lgb_params(self,space):
        lgb_params = dict()
        lgb_params['epoch'] =  10
        lgb_params['lambda'] =  space['lambda']
        lgb_params['lr'] =  space['lr']
        lgb_params['task'] = self.task
        lgb_params['bayesian'] = False
        lgb_params['patience'] = 50
        lgb_params['cuda'] = self.cuda
        lgb_params['device'] = self.device
        self.data_obj.ratio =space['ratio']
        self.data_obj.chunks = self.chunks
        return lgb_params

    def init_train(self,params):
        if not self.bayesian:
            model = linear_regression(self.nr_cols).to(self.device)
        else:
            model =bayesian_linear_regression(self.nr_cols).to(self.device)
        for n,p in model.named_parameters():
            print(n)
            print(p.shape)
            print(p.requires_grad)
        opt = torch.optim.Adam(model.parameters(), params['lr'])
        lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=params['patience'], factor=0.5)
        loss_func = get_loss_func(params)
        for i in range(params['epoch']):
            for j in range(self.its):
                s = time.time()
                self.data_obj.set_mode('train')
                X, y = self.data_obj.get_batch()
                if self.cuda:
                    X = X.to(self.device)
                    y = y.to(self.device)
                if not self.bayesian:
                    y_pred,reg = model(X)
                    pred_loss = loss_func(y_pred,y)
                else:
                    middle_term,last_term,reg = model(X)
                    pred_loss = loss_func(y,middle_term,last_term)
                l =  pred_loss + reg*params['lambda']
                opt.zero_grad()
                l.backward()
                opt.step()
                lrs.step(l)
                e = time.time()
                print(e-s)
            print(f'train error: {pred_loss.data}')

        val = calculate_loss_no_grad(model, dataloader=self.data_obj, train_config=params, task=self.task, mode='val')
        test = calculate_loss_no_grad(model, dataloader=self.data_obj, train_config=params, task=self.task, mode='test')

        return val.data,test.data

    def __call__(self, params):
        lgb_params = self.get_lgb_params(params)
        start = time.time()
        val_loss,test_loss = self.init_train(lgb_params)
        end = time.time()
        print(end-start)
        return {'loss': -val_loss, 'status': STATUS_OK,'test_loss': -test_loss}

    def run(self):
        self.trials = Trials()
        best = hyperopt.fmin(fn=self,
                             space=self.space,
                             algo=tpe.suggest,
                             max_evals=self.hyperits,
                             trials=self.trials,
                             verbose=3)
        print(space_eval(self.space, best))
        pickle.dump(self.trials,
                    open(self.save_path + '/' + self.name + '.p',
                         "wb"))