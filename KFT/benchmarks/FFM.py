from hyperopt import hp,tpe,fmin,Trials,STATUS_OK,space_eval
import hyperopt
import  sklearn.metrics as metrics
from KFT.util import get_free_gpu
import os
import time
import pickle
from KFT.benchmarks.utils import read_benchmark_data,get_auc,get_R_square
import torch
from KFT.job_utils import get_loss_func,calculate_loss_no_grad
from torchfm.model.ffm import FieldAwareFactorizationMachineModel

class FFM(FieldAwareFactorizationMachineModel):
    def __init__(self,field_dims, embed_dim):
        super(FFM, self).__init__(field_dims, embed_dim)

    def get_reg_term(self):
        reg = 0
        for p in self.parameters():
            reg+= p**2
        return reg

    def forward(self, x):
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return x.squeeze(1),self.get_reg_term()

class xl_FFM():
    def __init__(self,seed,y_name,data_path,save_path,params):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.data_obj = read_benchmark_data(y_name=y_name, path=data_path, seed=seed)
        self.nr_cols = self.data_obj.nr_cols*[1]
        self.hyperits = params['hyperopts']
        self.task = 'reg' if params['regression'] else 'binary'
        self.chunks= params['chunk']
        self.its = params['its']
        self.name = f'FFM_{seed}'
        self.cuda=params['cuda']
        if self.cuda:
            gpu = get_free_gpu(10)[0]
            self.device = f'cuda:{gpu}'
        else:
            self.device = f'cpu'
        self.space = {
            'k': hp.quniform('k', 4, 20, 1),
            'lambda': hp.uniform('lambda', 0.0, 0.005),
            'lr': hp.uniform('lr', 0.05, 1.0),
            'ratio': hp.uniform('ratio', 1e-6, 0.1),
        }

    def get_lgb_params(self,space):
        lgb_params = dict()
        lgb_params['epoch'] =  10
        lgb_params['k'] =  int(space['k'])
        lgb_params['lambda'] =  space['lambda']
        lgb_params['lr'] =  space['lr']
        lgb_params['nthread'] = 30
        lgb_params['task'] = self.task
        lgb_params['bayesian'] = False
        lgb_params['patience'] = 20
        self.data_obj.ratio =space['ratio']
        self.data_obj.chunks = self.chunks
        return lgb_params

    def init_train(self,params):
        model = FFM(self.nr_cols,params['k']).to(self.device)
        opt = torch.optim.Adam(model.parameters(), params['lr'])
        lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, factor=0.5)
        loss_func = get_loss_func(params)
        for i in range(params['epoch']):
            for j in range(self.its):
                self.data_obj.set_mode('train')
                X, y = self.data_obj.get_batch()
                if self.cuda:
                    X = X.to(self.device)
                    y = y.to(self.device)
                y_pred,reg = model(X)
                l = loss_func(y_pred,y)
                opt.zero_grad()
                l.backward()
                if params['partience']:
                    val = calculate_loss_no_grad(model,dataloader=self.data_obj,train_config=params,task=self.task,mode='val')
                    lrs.step(val)
                    print(f'ref met: {val.data}')
                    print(f'train error: {l.data}')
        val = calculate_loss_no_grad(model, dataloader=self.data_obj, train_config=params, task=self.task, mode='val')
        test = calculate_loss_no_grad(model, dataloader=self.data_obj, train_config=params, task=self.task, mode='test')

        return val.data,test.data

    def __call__(self, params):
        lgb_params = self.get_lgb_params(params)
        start = time.time()
        val_loss,test_loss = self.init_train(lgb_params)
        end = time.time()
        print(end-start)
        return {'loss': val_loss, 'status': STATUS_OK,'test_loss': test_loss}

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

