import torch

class CMEE_MODEL(torch.nn.Module):
    
    def __init__(self,data_extractor,shared_model,head1,head2,thresholding,contextual_embeddings = None,word_embeddings = None,device = None):
        super().__init__()
        assert(contextual_embeddings != None or word_embeddings != None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == None else torch.device(device)
        self.data_extractor = data_extractor
        self.contextual_embeddings = contextual_embeddings.to(self.device) if contextual_embeddings != None else None
        self.word_embeddings = word_embeddings.to(self.device) if word_embeddings != None else None
        self.model = shared_model.to(self.device)
        self.head1 = head1.to(self.device)
        self.head2 = head2.to(self.device)
        self.thresholding = thresholding
        
    def forward(self,x):
        if self.training == False and type(x) == str:
            file_no = x
            x = self.data_extractor.extract(file_no,False)
            
        if self.contextual_embeddings != None:
            e1 = self.contextual_embeddings(x["sentences"])
        if self.word_embeddings != None:
            e2 = self.word_embeddings(x["sentences"])
        
        if self.word_embeddings != None and self.contextual_embeddings != None:
            o = torch.cat([e1,e2],axis = 1)
        elif self.contextual_embeddings != None:
            o = e1
        else:
            o = e2
            
        y = self.model(o)
        return (self.head1(y),self.head2(y),x)