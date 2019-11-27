from KFT.benchmarks.lgbm import lgbm

if __name__ == '__main__':
    params= dict()
    params['its'] = 1000
    params['hyperopts'] = 20
    params['regression'] = True
    SAVE_PATH = './lgbm_private'
    y_name = 'total_sales'
    data_path = 'benchmark_data_lgbm'
    for i in [1,2,3,4,5]:
        l = lgbm(seed=seed,y_name=y_name,data_path=data_path,save_path=SAVE_PATH,params=params)
        l.run()
