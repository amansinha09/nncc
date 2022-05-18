import torch

class wCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction = 'none')
        
    def forward(self,logits,labels):
        loss = self.ce(logits,labels)
        dlogits = torch.nn.functional.softmax(logits.detach(),dim = -1)
        return torch.sum(loss * ((1 - torch.gather(dlogits,-1,labels.reshape(-1,1)))**0.1))
    
class LossCMEE():
    def __init__(self,k1,k2,loss_fn = None):
        self.k1 = k1
        self.k2 = k2
        self.loss_fn = wCrossEntropyLoss() if loss_fn == None else loss_fn
        
    def __call__(self,out1,out2,data):
        out1,out2 = out1,out2
        k1,k2 = self.k1,self.k2
        loss1 = torch.nan_to_num(self.loss_fn(out1,data["labels"]))
        o1,o2,o3,o4,o5 = out2[:,:7],out2[:,7:10],out2[:,10:14],out2[:,14:18],out2[:,18:]
        inds = torch.where(torch.logical_or(data["labels"] == 1,data["labels"] == 2 ))[0]
        loss2 = torch.nan_to_num(self.loss_fn(o1[inds],data["attribute_labels"][:,0][inds].reshape(-1))
                               + self.loss_fn(o2[inds],data["attribute_labels"][:,1][inds].reshape(-1))
                               + self.loss_fn(o3[inds],data["attribute_labels"][:,2][inds].reshape(-1))
                               + self.loss_fn(o4[inds],data["attribute_labels"][:,3][inds].reshape(-1))
                               + self.loss_fn(o5[inds],data["attribute_labels"][:,4][inds].reshape(-1)))
        return k1*loss1 + k2*loss2

class Linear_Head(torch.nn.Module):
    def __init__(self,size_in,size_out,dropout = 0.5):
        
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(size_in , size_out)
        
    def forward(self,x):
        x = self.dropout(x)
        return self.linear(x)

class Linear_and_LSTM_Head(torch.nn.Module):
    def __init__(self,size_in,size_out,dropout = 0.5):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        
        self.dropout = torch.nn.Dropout(dropout)
        self.bidirectional_LSTM = torch.nn.LSTM(size_in,size_out,batch_first = True,bidirectional = True)
        self.linear_merge_predictions = torch.nn.Linear( 3 * size_out,size_out )
        self.linear = torch.nn.Linear(size_in,size_out)
        
    def forward(self,x):
        x = self.dropout(x)
        o1,_ = self.bidirectional_LSTM(x.detach())
        o1 = o1.reshape( -1 , self.size_out * 2 )
        o2 = self.linear(x)
        return self.linear_merge_predictions(torch.cat([o1,o2],dim = -1))