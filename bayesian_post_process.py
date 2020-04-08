from KFT.util import post_process,plot_VI


if __name__ == '__main__':
    PATH = './report_movielens_multivariate_10m_split_2/'
    # plot_VI(PATH,['idx_0','idx_2'],seed=3)
    post_process(PATH,'test_loss',reverse=False,bayesian=True)