import torch

class linear_regression(torch.nn.Module):
    def __init__(self,nr_col):
        super(linear_regression, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(nr_col),requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(1),requires_grad=True)

    def forward(self, X):
        return X@self.w+self.b, self.w**2

class bayesian_linear_regression(linear_regression):
    def __init__(self, nr_col):
        super(bayesian_linear_regression, self).__init__(nr_col)
        self.w_sigma = torch.nn.Parameter(torch.randn(nr_col),requires_grad=True)
        self.b_sigma = torch.nn.Parameter(torch.randn(nr_col),requires_grad=True)

    def KL(self,mean,sig):
        return torch.mean(0.5*(sig.exp()+mean**2-sig-1))

    def forward(self,X):
        a = X@self.w
        middle_term = a +self.b
        last_term = a**2 + 2*self.b*a+X**2@self.w_sigma.exp()+self.b_sigma.exp()+self.b**2
        KL_tot = self.KL(self.w,self.w_sigma)+self.KL(self.b,self.b_sigma)
        return middle_term,last_term,KL_tot
