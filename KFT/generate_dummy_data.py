import pandas as pd
import numpy as np
import sklearn.datasets as ds
import torch

np.random.seed(9001)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_int_dates(x):
    y = x.split('-')
    return list(map(lambda z: int(z), y))

def generate_time(days = 400):
    times = pd.to_datetime(get_time(delta=days)).astype('str')
    time_data = torch.Tensor(list(map(get_int_dates, times)))
    return time_data
def get_sales_data(number_of_articles = 100):
    cov = ds.make_spd_matrix(number_of_articles, random_state=None)*20
    mean = cov.diagonal()
    X = np.random.multivariate_normal(mean, cov ,1700)
    return X


def get_time(y=2013,m=1,d=1,delta=1500):
    times = pd.date_range(str(y)+'-'+str(m)+'-'+str(d),
                          periods=delta,
                          freq='1d')
    time = np.array(times)
    return time

def get_timeseries(number_of_articles):
    X = get_sales_data(number_of_articles)
    X = X.clip(min=0)
    X = np.round(X)
    column_data = []
    T = get_time()
    _t_len = len(T)

    for i in range(X.shape[1]):
        N = np.random.randint(30,1400)
        _data = X[:,i]
        _timeseries = np.random.choice(_data,N,replace=False)
        _values_vector = np.zeros(_t_len)
        _values_vector[:] = np.nan
        start_index = np.random.randint(0,_t_len-N)
        _values_vector[start_index:start_index+N] = _timeseries
        column_data.append([T.tolist(),_values_vector.tolist()])
    return column_data

def make_data(article_names,cities=['Oxford','Cambridge']):
    df = pd.DataFrame(columns=['cities','article','timestamp','sales_val'])

    for c in cities:
        timeseries_data = get_timeseries(len(article_names))
        for i in range(len(article_names)):
            n = len(timeseries_data[i][0])
            _data =  [[c,article_names[i],pd.to_datetime(timeseries_data[i][0][l]),timeseries_data[i][1][l] ] for l in range(n)]
            append_df = pd.DataFrame( _data ,columns=['cities','article','timestamp','sales_val'])
            df = df.append(append_df)
    return df
if __name__ == "__main__":
    number_of_articles = 50
    cities = [
        "Assen", "Almere", "Leeuwarden", "Nijmegen", "Groningen",
        "Maastricht", "Eindhoven", "Amsterdam", "Enschede", "Utrecht",
        "Middelburg", "Rotterdam",
        "Fredrikstad", "Baerum", "Oslo", "Ringsaker", "Gjovik",
        "Drammen", "Sandefjord", "Skien", "Arendal", "Kristiansand",
        "Stavanger", "Bergen", "Forde", "Alesund", "Trondheim",
        "Stjordal", "Bodo", "Tromso", "Alta", "Stockholm", "Malmö", "Göteborg"
    ]
    article_names = list(set([str(np.random.randint(100000, 999999)) for i in range(number_of_articles)]))
    data = make_data(article_names,cities)
    # print(data)
    data.to_csv('generated_data.csv')

    """
    Generate base estimator data
    """

    # time_element = generate_time(days=1500)
    # n = time_element.shape[0]
    # model = matrix_kernel_factorization([34,50,n], 10, time_element)
    # model = model.to(device)
    # _model_data = model.full_forward()
    # noise_tensor = torch.randn(*_model_data.shape).to(device)
    # model_data = _model_data + noise_tensor*2
    # torch.save(model_data,'./toy_data_bigger.p')

    #TODO: Do benchmarking for linear model, xgboost and neuralnet. Also implement test and training context


