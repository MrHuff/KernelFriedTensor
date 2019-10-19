import torch

class TT_component(torch.nn.Module):
    def __init__(self,r_1,n_list,r_2):
        super(TT_component, self).__init__()
        self.n_len = len(n_list)
        self.shape_list  = [n_i for n_i in n_list].insert(0,r_1).append(r_2)
        self.TT_core = torch.nn.Parameter(torch.randn_like(*self.shape_list),requires_grad=True)

    def forward(self,indices):
        batch = self.TT_core.index_select(dim=1,index=indices[:,0])
        for i in range(self.n_len-1): #Implictily recurse if there are more dimensions than 1
            batch = batch.index_select(dim=2+i,index=indices[i,1+i])
        return batch

class TT_kernel_component(TT_component):
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_params):
        super(TT_kernel_component, self).__init__(r_1,n_list,r_2)
        

