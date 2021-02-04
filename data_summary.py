import torch
import pickle
import pandas as pd
datasets = ['public_data_t_fixed/',
            'public_movielens_data_t_fixed/',
            'tensor_data_t_fixed/',
            'electric_data/',
            'CCDS_data/',
            'traffic_data/']
names = ['Alcohol Sales',
         'Movielens-20M',
         'Fashion Retail Sales',
         'Electric',
         'CCDS',
         'Traffic'
         ]
if __name__ == '__main__':
    df_data = []
    for d,name in zip(datasets,names):
        X,y = torch.load(f'{d}all_data.pt')
        original_shape = list(pickle.load(open(d + 'full_tensor_shape.pickle', 'rb')))
        full_str = r'\times '.join([str(elem) for elem in original_shape])
        print(full_str)
        row = [name,round(y.mean().item(),3),round(y.std().item(),3),y.shape[0],'$'+full_str+'$']
        df_data.append(row)
    df_data = pd.DataFrame(df_data,columns=['Dataset','Mean','Std.Dev','N','Dimensionality'])
    df_data = df_data.set_index('Dataset')
    print(df_data.to_latex(escape=False))