from hyperopt import hp,tpe,fmin,Trials,STATUS_OK,space_eval
import hyperopt
from sklearn.metrics.regression import mean_squared_error
import  sklearn.metrics as metrics
import os
import time
import pickle
from KFT.benchmarks.utils import read_benchmark_data
import lightgbm
import numpy as np


def get_auc(Y,y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(Y, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
def get_R_square(Y,y_pred):
    mse = mean_squared_error(Y,y_pred)
    var = np.var(Y)
    return -(1-mse/var)

class lgbm():
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
        self.D_train = lightgbm.Dataset(X_train, y_train, categorical_feature='auto')
        self.D_val = lightgbm.Dataset(self.X_val, self.y_val, categorical_feature='auto')
        self.hyperits = params['hyperopts']
        self.task = 'regression' if params['regression'] else 'cross_entropy'
        self.train_objective = 'mse' if params['regression'] else 'cross_entropy'
        self.eval_objective = get_R_square if params['regression'] else get_auc
        self.name = f'{self.task}_lgbm_{seed}'
        self.its = params['its']
        self.space = {
            'num_leaves': hp.quniform('num_leaves', 1, 200, 1),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 30, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
            'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
            'max_bin': hp.quniform('max_bin', 128, 512, 1),
            'bagging_freq': hp.quniform('bagging_freq', 1, 5, 1),
            'lambda_l1': hp.uniform('lambda_l1', 0, 10),
            'lambda_l2': hp.uniform('lambda_l2', 0, 10),
        }

    def get_lgb_params(self,space):
        lgb_params = dict()
        lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
        lgb_params['application'] = self.task
        lgb_params['metric'] = 'mse'
        lgb_params['num_class'] = 1
        lgb_params['learning_rate'] = space['learning_rate']
        lgb_params['num_leaves'] = int(space['num_leaves'])
        lgb_params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
        lgb_params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
        lgb_params['max_depth'] = -1
        lgb_params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
        lgb_params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
        lgb_params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 256
        lgb_params['feature_fraction'] = space['feature_fraction']
        lgb_params['bagging_fraction'] = space['bagging_fraction']
        lgb_params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1
        return lgb_params

    def __call__(self, params):
        lgb_params = self.get_lgb_params(params)
        start = time.time()
        model = lightgbm.train(lgb_params,
                               self.D_train,
                               num_boost_round=self.its,
                               valid_sets=self.D_val,
                               early_stopping_rounds=100,
                               verbose_eval=True,
                               )
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

