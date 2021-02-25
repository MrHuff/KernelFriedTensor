import numpy as np
import matplotlib.pyplot as plt


def load_and_concat(path,file):
    data_container = []
    min_lengths = np.inf
    for i in range(1,6):
        vec = np.load(path+f'/{file}_{i}.npy')
        data_container.append(vec)
        length = vec.shape[0]
        if length<min_lengths:
            min_lengths = length
    # for i in range(len(data_container)):
    #     data_container[i] = data_container[i][:min_lengths]

    return np.stack(data_container)
def mean_std(vec):
    mean = vec.mean(axis=0)
    std = vec.std(axis=0)
    ci_neg = mean-std
    ci_pos = mean+std
    # ci_neg = vec.min(axis=0)
    # ci_pos = vec.max(axis=0)
    x = np.arange(1,mean.shape[0]+1)
    return mean,ci_neg,ci_pos,x

def plot_point(save_name,errors,title):
    path = 'KFT_motivation'
    path_2 = 'KFT_motivation_old_setup'
    vec = load_and_concat(path,errors)
    # for el in vec:
    #     plt.plot(el,color='b')
    mean,ci_neg,ci_pos,x = mean_std(vec)
    plt.plot(x,mean,label='KFT')
    plt.fill_between(x,ci_neg,ci_pos,color='b',alpha=0.1)
    vec_2 = load_and_concat(path_2, errors)
    # for el in vec_2:c
    #     plt.plot(el,color='r')
    mean, ci_neg, ci_pos, x = mean_std(vec_2)
    plt.plot(x, mean,label='Naive')
    plt.fill_between(x, ci_neg, ci_pos, color='r', alpha=0.1)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{save_name}.png')
    plt.clf()

if __name__ == '__main__':
    plot_point('train','train_errors','Training error')
    plot_point('val','val_errors','Validation error')
    plot_point('test','test_errors','Test error')

    # vec = load_and_concat(path,'val_errors')
    # plt.plot(vec.transpose())
    # plt.savefig('val.png')
    # plt.clf()
    # vec = load_and_concat(path,'test_errors')
    # plt.plot(vec.transpose())
    # plt.savefig('test.png')
    # plt.clf()
