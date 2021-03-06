from KFT.benchmarks.lgbm import lgbm

def run():
    GPU = True
    params= dict()
    params['gpu'] = GPU
    params['its'] = 1000
    params['hyperopts'] = 20
    params['regression'] = True
    params['num_threads'] = 0
    SAVE_PATH = './lgbm_movielens_GPU'
    y_name = 'rating'
    data_path = 'movielens_parquet'
    for i in [1,2,3,4,5]:
        l = lgbm(seed=i,y_name=y_name,data_path=data_path,save_path=SAVE_PATH,params=params)
        l.run()

    params= dict()
    params['gpu'] = GPU
    params['its'] = 1000
    params['hyperopts'] = 20
    params['regression'] = True
    params['num_threads'] = 0
    SAVE_PATH = './lgbm_alcohol_GPU'
    y_name = 'Bottles Sold'
    data_path = 'alcohol_sales_parquet'
    for i in [1,2,3,4,5]:
        l = lgbm(seed=i,y_name=y_name,data_path=data_path,save_path=SAVE_PATH,params=params)
        l.run()

    params= dict()
    params['gpu'] = GPU
    params['its'] = 1000
    params['hyperopts'] = 20
    params['regression'] = True
    params['num_threads'] = 0
    SAVE_PATH = './private_lgbm_GPU'
    y_name = 'total_sales'
    data_path = 'benchmark_data_lgbm'
    for i in [1,2,3,4,5]:
        l = lgbm(seed=i,y_name=y_name,data_path=data_path,save_path=SAVE_PATH,params=params)
        l.run()
if __name__ == '__main__':
    run()