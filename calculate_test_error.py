from KFT.util import get_test_errors


if __name__ == '__main__':
    data_path = './report_movielens_data_ml-10m/all_data.pt'
    get_test_errors('./report_movielens_univariate_10m_split_2/','test_loss',data_path,split_mode=2,reverse=False,bayes=True,metric_name='test_loss_final')