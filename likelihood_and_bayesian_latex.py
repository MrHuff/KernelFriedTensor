from plot_bayesian_forecasts import *
import torch
import pandas as pd

def get_likelihood(j_tmp,best_ind):
    j_tmp.load_dumped_model(best_ind + 1)
    j_tmp.model.turn_on_all()
    j_tmp.model.to(j_tmp.device)
    j_tmp.init_dataloader(0.01)
    total_cal_error_test, test_cal_dict, predictions, test_likelihood, test_ELBO = j_tmp.calculate_calibration(
        mode='test', task=j_tmp.train_config['task'])
    return test_ELBO,test_likelihood


def find_p_file(path):
    files = os.listdir(path)
    for el in files:
        if el[-2:]=='.p':
            return el

def save_likelihood_metrics(folder,folder_2,job_indices):
    if not os.path.exists(folder+'/all_results.csv'):
        trial_list = []
        columns = ['dataset','model','$\Xi_{0.05}$','$\Xi_{0.15}$','$\Xi_{0.25}$','$\Xi_{0.35}$','$\Xi_{0.45}$','$\Xi$','$R^2$','$\eta_{\text{criteria}}$','ELBO','log-likelihood']
        for job_ind in job_indices:
            path_a = folder + f'/job_{job_ind}'
            fname = find_p_file(path_a)
            trials = pickle.load(open(folder + f'/job_{job_ind}/{fname}', "rb"))
            best_trial = sorted(trials.trials, key=lambda x: x['result']['test_loss'], reverse=False)[0]
            best_tid = best_trial['tid']
            key_init = load_obj(f'job_{job_ind}.pkl', f"{folder_2}/")
            j_tmp = job_object(key_init)
            j_tmp.save_path = f'{folder}/job_{job_ind}'
            ELBO,likelihood = get_likelihood(j_tmp,best_tid)
            best_trial['test_ELBO'] = ELBO
            best_trial['test_likelihood'] = likelihood
            trial_list.append([folder_2,folder_2,
                               best_trial['result']['test_cal_dict'][5],
                               best_trial['result']['test_cal_dict'][15],
                               best_trial['result']['test_cal_dict'][25],
                               best_trial['result']['test_cal_dict'][35],
                               best_trial['result']['test_cal_dict'][45],
                               best_trial['result']['test_loss']+best_trial['result']['test_loss_final']['R2'],
                               best_trial['result']['test_loss_final']['R2'],
                               best_trial['result']['test_loss'],
                               best_trial['test_ELBO'],
                               best_trial['test_likelihood'],
                               ])
        df = pd.DataFrame(trial_list,columns=columns)
        df.to_csv(folder+'/all_results.csv')


if __name__ == '__main__':
    columns = ['dataset', 'model', '$\Xi_{0.05}$', '$\Xi_{0.15}$', '$\Xi_{0.25}$', '$\Xi_{0.35}$', '$\Xi_{0.45}$',
               '$\Xi$', '$R^2$', '$\eta_{\text{criteria}}$', 'ELBO', 'log-likelihood']
    tex_final_name = 'retail_bayesian_results'
    folder_2_list = ['retail_benchmark_bayesian',
                     'retail_20_bayesian_dual_univariate_LS',
                     'retail_20_bayesian_dual_multivariate_LS',
                     'retail_20_bayesian_dual_multivariate',
                     'retail_20_bayesian_dual_univariate',
                     ]
    job_indices = [0, 1, 2]
    summary_df_list = []
    for folder_2 in folder_2_list:
        folder = f'{folder_2}_results'
        save_likelihood_metrics(folder,folder_2,job_indices)
        if os.path.exists(folder+'/all_results.csv'):
            df = pd.read_csv(folder+'/all_results.csv',index_col=0)
            summary = df.describe().transpose()
            tmp_df = summary[['mean','std']]
            tmp_df['latex_col'] = tmp_df['mean'].apply(lambda x: '$'+str(round(x,3))+'\pm ') + tmp_df['std'].apply(lambda x: str(round(x,3))+'$')
            tmp_df = tmp_df['latex_col'].tolist()
            tmp_df.insert(0,folder)
            tmp_df.insert(0,folder)
            summary_df_list.append(tmp_df)
    df = pd.DataFrame(summary_df_list,columns=columns)
    df.to_latex(f'{tex_final_name}.tex',escape=False,index=None)
