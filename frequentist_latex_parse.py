import torch
import pandas as pd
import os
import pickle

def find_p_file(path):
    files = os.listdir(path)
    for el in files:
        if el[-2:]=='.p':
            return el

def parse_folder_frequentist(folder, folder_2, job_indices):
    if not os.path.exists(folder + '/all_results.csv'):
        trial_list = []
        columns = ['dataset', 'model', '$R^2$','NRSME','ND']
        for job_ind in job_indices:
            path_a = folder + f'/job_{job_ind}'
            fname = find_p_file(path_a)
            trials = pickle.load(open(folder + f'/job_{job_ind}/{fname}', "rb"))
            best_trial = sorted(trials.trials, key=lambda x: x['result']['test_loss'], reverse=True)[0]
            print(best_trial)
            trial_list.append([folder_2,folder_2,
                               best_trial['result']['other_test']['R2'],
                               best_trial['result']['other_test']['NRMSE'],
                               best_trial['result']['other_test']['ND'],
                               ])
        df = pd.DataFrame(trial_list, columns=columns)
        df.to_csv(folder + '/all_results.csv')

if __name__ == '__main__':
    columns = ['dataset', 'model', '$R^2$', 'RSME', 'ND']
    tex_final_name = 'alcohol_results'
    folder_2_list = [
        'alcohol_benchmark'
                     ]
    job_indices = [0, 1, 2,3,4]
    summary_df_list = []
    for folder_2 in folder_2_list:
        folder = f'{folder_2}_results'
        parse_folder_frequentist(folder,folder_2,job_indices)
        df = pd.read_csv(folder + '/all_results.csv', index_col=0)
        summary = df.describe().transpose()
        tmp_df = summary[['mean', 'std']]
        tmp_df['latex_col'] = tmp_df['mean'].apply(lambda x: '$' + str(round(x, 3)) + '\pm ') + tmp_df['std'].apply(
            lambda x: str(round(x, 3)) + '$')
        tmp_df = tmp_df['latex_col'].tolist()
        tmp_df.insert(0, folder)
        tmp_df.insert(0, folder)
        summary_df_list.append(tmp_df)

    if not os.path.exists(f'{tex_final_name}.tex'):
        df = pd.DataFrame(summary_df_list,columns=columns)
        df.to_latex(f'{tex_final_name}.tex',escape=False,index=None)
