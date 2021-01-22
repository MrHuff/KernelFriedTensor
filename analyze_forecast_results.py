import pandas as pd
from generate_parameters import load_obj
import os
import pickle
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 24)

folder_2 = "jobs_traffic_parallel"
list_of_stuff = os.listdir(folder_2)


folder = f"{folder_2}_results"
list_of_res = os.listdir(folder)


if __name__ == '__main__':
    cat = []
    for i,el in enumerate(list_of_stuff):
        keystr = el
        key_load = load_obj(keystr,f"{folder_2}/")
        w_val = key_load['lambda_W_a']
        T_val = key_load['lambda_T_x_a']
        fold = key_load['temporal_folds'][0]
        df_tmp = pd.read_csv(folder+f'/job_{i}/test_df.csv',index_col=0)
        trials  = pickle.load(open(folder+f'/job_{i}/frequentist_0.p', "rb"))
        R_val = trials.trials[0]['misc']['vals']['R'][0]+100
        print(key_load)
        df_tmp['job_ind'] = f'job_{i}'
        df_tmp['file_name'] = keystr
        df_tmp['w']=w_val
        df_tmp['T']=T_val
        df_tmp['fold']=fold
        df_tmp['R']=R_val
        df_tmp['lambda_W'] = trials.trials[0]['misc']['vals']['lambda_W'][0]
        df_tmp['lambda_T_x'] = trials.trials[0]['misc']['vals']['lambda_T_x'][0]
        df_tmp['bs_ratio'] = trials.trials[0]['misc']['vals']['batch_size_ratio'][0]
        df_tmp['max_R'] = key_load['max_R']
        cat.append(df_tmp)
    df = pd.concat(cat,axis=0)
    df = df.sort_values(['fold','NRMSE'])
    print(df)
    df.to_csv(f"analysis_{folder_2}.csv")


