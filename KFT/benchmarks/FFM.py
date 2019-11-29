from hyperopt import hp,tpe,fmin,Trials,STATUS_OK,space_eval
import hyperopt
from sklearn.metrics.regression import mean_squared_error
import  sklearn.metrics as metrics
import os
import time
import pickle
from KFT.benchmarks.utils import read_benchmark_data,get_auc,get_R_square
import xlearn as xl
import numpy as np

class xl_FFM():
    def __init__(self,seed,y_name,data_path,save_path,params):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        data_obj = read_benchmark_data(y_name=y_name, path=data_path, seed=seed)
        X_train = data_obj.X_train
        y_train = data_obj.Y_train
        self.X_val = data_obj.X_val
        self.y_val = data_obj.Y_val
        self.X_test = data_obj.X_test
        self.y_test = data_obj.Y_test
        self.D_train = xl.DMatrix(X_train, y_train)
        self.D_val = xl.DMatrix(self.X_val, self.y_val) #needs to be OHE
        self.hyperits = params['hyperopts']
        self.task = 'reg' if params['regression'] else 'binary'
        self.train_objective = 'rmse' if params['regression'] else 'auc'
        self.eval_objective = get_R_square if params['regression'] else get_auc
        self.name = f'{self.task}_lgbm_{seed}'
        self.its = params['its']
        self.space = {
            'k': hp.quniform('k', 4, 20, 1),
            'lambda': hp.uniform('lambda', 0.0, 0.005),
            'lr': hp.uniform('lr', 0.05, 1.0),
        }

    def get_lgb_params(self,space):
        lgb_params = dict()
        lgb_params['opt'] =  'adagrad'
        lgb_params['epoch'] =  10
        lgb_params['stop_window'] =  3
        lgb_params['k'] =  int(space['k'])
        lgb_params['lambda'] =  space['lambda']
        lgb_params['lr'] =  space['lr']
        lgb_params['nthread'] = 30
        lgb_params['task'] = self.task
        lgb_params['metric'] = self.train_objective

        return lgb_params

    def __call__(self, params):
        lgb_params = self.get_lgb_params(params)
        start = time.time()
        model = xl.create_ffm()
        model.setTrain(self.D_train)  # Training data
        model.setValidate(self.D_val)
        model.fit(lgb_params,self.save_path)
        end = time.time()
        print(end-start)
        nb_trees = model.best_iteration
        print('nb_trees={}'.format(nb_trees))
        val_loss = self.get_eval_score(model,nb_trees,self.X_val,self.y_val)
        test_loss = self.get_eval_score(model,nb_trees,self.X_test,self.y_test)

        return {'loss': val_loss, 'status': STATUS_OK,'test_loss': test_loss}

    def get_eval_score(self,model,nb_trees,X,y):
        preds = model.predict(X,num_iteration=nb_trees)
        return self.eval_objective(y,preds)

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

