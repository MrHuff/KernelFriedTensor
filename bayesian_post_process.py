from KFT.util import post_process,plot_VI


if __name__ == '__main__':
    PATH = './public_job_arch_0_dual_bayesian_univariate/'
    plot_VI(PATH,['idx_0','idx_2'])
    post_process(PATH,'test_loss_final',reverse=True)