import numpy as np
import torch

class piecewise_cross_entropy_loss():
    def __init__(self,k1=1,k2=1):
        self.k1 = k1
        self.k2 = k2
    
    def __call__(self,out1,out2,data):
        k1,k2 = self.k1,self.k2
        loss1 = torch.nan_to_num(torch.nn.functional.cross_entropy(out1,torch.tensor(data["labels"],dtype = torch.int64).to("cuda")))
        o1,o2,o3,o4,o5 = out2[:,:7],out2[:,7:10],out2[:,10:14],out2[:,14:18],out2[:,18:]
        inds = torch.where(torch.tensor(np.logical_or(data["labels"] == 1,data["labels"] == 2 )))[0]
        loss2 = torch.nan_to_num(torch.nn.functional.cross_entropy(o1[inds],torch.tensor(data["attribute_labels"][:,0][inds],dtype = torch.int64).reshape(-1).to("cuda"))
                               + torch.nn.functional.cross_entropy(o2[inds],torch.tensor(data["attribute_labels"][:,1][inds],dtype = torch.int64).reshape(-1).to("cuda"))
                               + torch.nn.functional.cross_entropy(o3[inds],torch.tensor(data["attribute_labels"][:,2][inds],dtype = torch.int64).reshape(-1).to("cuda"))
                               + torch.nn.functional.cross_entropy(o4[inds],torch.tensor(data["attribute_labels"][:,3][inds],dtype = torch.int64).reshape(-1).to("cuda"))
                               + torch.nn.functional.cross_entropy(o5[inds],torch.tensor(data["attribute_labels"][:,4][inds],dtype = torch.int64).reshape(-1).to("cuda")))
        return k1*loss1 + k2*loss2
