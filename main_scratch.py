
from KFT.job_utils import parse_args,job_object #parse_args = passera argument classen, job_object="kärnan; instansierar modell och kör den"


if __name__ == '__main__':

    _configs = [
                {'a':1,'word':'wl','latent_scale':False,'special_mode':0,'delete_side_info':[0,1,2],'old_setup':False,'R':5},
               ]
    for c in _configs:
      a = c['a']
      word = c['word']
      latent_scale = c['latent_scale']
      special_mode = c['special_mode']
      delete_side_info = c['delete_side_info']
      old_setup = c['old_setup']
      R = c['R']
      def run_func_db(param):
          seed = param['x']
          job_path = './mathias/' #Här är din data, den har "index data" + sid info
          save_path = f'./test_job/' #Här sparas all data
          params = {'PATH':job_path, #Här är alla viktiga parametrar
              'full_grad':False, #Ta hela tränings settet som batch, gör aldrig det här
              'latent_scale':latent_scale, #Flagga för LS-modell se papper
              'chunks':100, #För validering och test data chunks, rör ej generellt ta ett stort nummer om data size är stort
              'reg_para_a':0, #lower bound för lambda, regulariserings parametrar
              'reg_para_b':10, #upper bound
              'batch_size_a':0.005, # batch size %, 1.0 <-> full_grad=True, lower bound
              'batch_size_b':0.025, # batch size %, 1.0 <-> full_grad=True, upper bound
              'hyperits':20, #Antalet hyperiterationer
              'save_path':save_path, #self explanatory
              'task':'classification_acc', #reg = regression, classification_acc=accuracy, classification_auc
              'epochs':10, #Antalet epoker som körs
              'bayesian':False, #bayesian model, sätt till F
              'cuda':True, #Alltid True
              'sub_epoch_V':100, #Antalet its per epoch
              'seed':seed,
              'side_info_order':[0,1,2], #Index orningen av sido information
              'temporal_tag':[2], #Gör ingenting refaktorera
              'architecture':a, #cuda issues for architecture 1, consider uppgrading to cuda 10.1!, använt generellt 0, a =0
              'max_R':R, # Antalet latent:a faktorer, komplexiteten på din modell
              'max_lr':1e-1, #learning rate
              'old_setup':old_setup, #Rör ej alltid False, lägg inte till justerings faktorer
              'delete_side_info':delete_side_info, #Ta bort en del sidoinformation passera lista, None för ingenting
              'dual':False, #Dual/Primal mode
              'init_max':1e0, #Initialisering av vikter storhet
              'multivariate':False, #Rör ej
              'L':0, #alltid 0
              'factorize_latent':False, #Alltid False
              'mu_a':0, #rör ej
              'mu_b':0, #rör ej
              'sigma_a':0, #rör ej
              'sigma_b':0, #rör ej
              'special_mode':special_mode, #Alltid 0
              'pos_weight':1.0,
              'kernels':['matern_1','matern_2','matern_3','rbf'] #Vilka kärnor vill du använda?
          }
          side_info,other_configs = parse_args(params)
          j = job_object(
          side_info_dict=side_info,
          configs=other_configs,
          seed=params['seed']
          )
          j.run_hyperparam_opt()
      run_func_db({'x':1})
