from KFT.benchmarks.linear_regression import linear_job_class
if __name__ == '__main__':
    args = dict()
    args['y_name'] = 'rating'
    args['data_path'] = 'movielens_parquet_hashed_scaled'
    args['SAVE_PATH'] = './linear_movielens_benchmark'
    args['bayesian'] = False
    args['cuda'] = False
    args['hyperopts'] = 20
    args['regression'] = True
    args['its'] = 100
    args['chunk'] = 50

    for  i in [1,2,3,4,5]:
        l = linear_job_class(seed=i,
                             y_name=args['y_name']
                             ,data_path=args['data_path']
                             ,save_path=args['SAVE_PATH']
                             ,params=args)
        l.run()

    args['y_name'] = 'Bottles Sold'
    args['data_path'] = 'alcohol_sales_parquet_hashed_scaled'
    args['SAVE_PATH'] = './linear_alcohol_benchmark'

    for  i in [1,2,3,4,5]:
        l = linear_job_class(seed=i,
                             y_name=args['y_name']
                             ,data_path=args['data_path']
                             ,save_path=args['SAVE_PATH']
                             ,params=args)
        l.run()


    args['y_name'] = 'total_sales'
    args['data_path'] = 'benchmark_data_private_hashed_scaled'
    args['SAVE_PATH'] = './private_linear_benchmark'

    for  i in [1,2,3,4,5]:
        l = linear_job_class(seed=i,
                             y_name=args['y_name']
                             ,data_path=args['data_path']
                             ,save_path=args['SAVE_PATH']
                             ,params=args)
        l.run()




