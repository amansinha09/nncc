import torch
import numpy as np
import torch_scatter

class Contextual_Encodings_preprocessor():
    
    def __init__(self,tokenizer,sentence_length,context_length,batch_size,cls_tk,sep_tk,pad_tk):
        self.tokenizer = tokenizer
        self.sentence_length = sentence_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.cls_tk = cls_tk
        self.sep_tk = sep_tk
        self.pad_tk = pad_tk
        
    def __call__(self,x):
        sentences = x
        repeats = np.zeros((len(sentences),),dtype = np.int64)
        for i in range(len(sentences)):
            repeats[i] = self.tokenizer.tokenize(sentences[i]).__len__()
        temporary = np.arange(0,len(sentences),dtype = np.int64)
        segment_ids = np.repeat(temporary,repeats)
        tokens = self.tokenizer.encode(sentences,is_split_into_words = True)[1:-1]
        assert(len(tokens) == len(segment_ids))
        segment_ids = np.pad(np.array(segment_ids),pad_width = (self.context_length,self.context_length + (self.sentence_length - (len(tokens) % self.sentence_length)) % self.sentence_length),constant_values = -1)
        tokens = np.pad(np.array(tokens),pad_width = (self.context_length,self.context_length + (self.sentence_length - (len(tokens) % self.sentence_length)) % self.sentence_length),constant_values = self.pad_tk)   
        tokens = np.array(tokens)
        segment_ids = np.array(segment_ids)
        _tokens = []
        _segment_ids =  []
        _attention_mask = []
        for i in range((len(tokens)-2*self.context_length)//self.sentence_length):
            temp_tk = np.concatenate([[self.cls_tk],tokens[i*self.sentence_length:2*self.context_length+(i+1)*self.sentence_length],[self.sep_tk]],axis = 0)
            temp_si = np.concatenate([[-1],segment_ids[i*self.sentence_length:2*self.context_length+(i+1)*self.sentence_length],[-1]],axis = 0)
            _tokens.append(np.array(temp_tk,dtype = np.int64))
            _segment_ids.append(np.array(temp_si,dtype = np.int64))
            _attention_mask.append((temp_tk!=self.pad_tk).astype(np.int64))
        data = torch.utils.data.TensorDataset(torch.tensor(_tokens),torch.tensor(_attention_mask))

        return {"dataloader":torch.utils.data.DataLoader(data,sampler = torch.utils.data.SequentialSampler(data),batch_size = self.batch_size),
        "segment_ids":torch.tensor(_segment_ids)}
        
class Contextual_Encodings_transformer(torch.nn.Module):
    
    def __init__(self,transformer):
        super().__init__()
        self.transformer = transformer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self,x):
        self.to(self.device)
        output_list = []
        if self.training:
            for tokens,attention_mask in x:
                output_list.append(self.transformer(tokens.to(self.device),attention_mask = attention_mask.to(self.device),token_type_ids = None).last_hidden_state)
            return torch.cat(output_list,axis = 0)
        else:
            with torch.no_grad():
                for tokens,attention_mask in x:
                    output_list.append(self.transformer(tokens.to(self.device),attention_mask = attention_mask.to(self.device),token_type_ids = None).last_hidden_state)
            return torch.cat(output_list,axis = 0)
        
class Contextual_Encodings_postprocessor(torch.nn.Module):
    
    def __init__(self,sentence_length,context_length,train_time_merge_strategy,test_time_merge_strategy,with_cls_embeddings):
        super().__init__()
        self.sentence_length = sentence_length
        self.context_length = context_length
        assert(test_time_merge_strategy.lower() in ["min","max","mean"])
        assert(train_time_merge_strategy.lower() in ["min","max","mean"])
        self.merge_strategy = [test_time_merge_strategy.lower(),train_time_merge_strategy.lower()]
        self.merge_fn = {"min":torch_scatter.scatter_min,
                   "max":torch_scatter.scatter_max,
                   "mean":torch_scatter.scatter_mean}
        self.with_cls_embeddings = with_cls_embeddings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self,x,segment_ids):
        embeddings = x[:,1 + self.context_length:1 + self.sentence_length + self.context_length,:]
        if self.with_cls_embeddings:
            embeddings = torch.cat([embeddings,torch.broadcast_to(x[:,0:1,:],embeddings.shape)],axis = -1)
        segment_ids = segment_ids[:,1 + self.context_length:1 + self.sentence_length + self.context_length]
        output = torch.zeros(1 + segment_ids.max(),embeddings.shape[-1])
        valid_indices = torch.where(segment_ids >= 0) 
        output = self.merge_fn[self.merge_strategy[self.training]](embeddings[valid_indices].to(self.device),segment_ids[valid_indices].to(self.device),dim = 0)
        return output

class Contextual_Representations_Model(torch.nn.Module):
    
    def __init__(self,transformer,tokenizer,sentence_length,context_length,batch_size,cls_tk,sep_tk,pad_tk,train_time_merge_strategy = "mean",test_time_merge_strategy = "mean",with_cls_embeddings = False):
        super().__init__()
        assert(sentence_length>=3 and context_length>=0)
        assert(sentence_length + 2 + 2 * context_length <= 512)
        
        self.preprocessor = Contextual_Encodings_preprocessor(tokenizer,sentence_length,context_length,batch_size,cls_tk,sep_tk,pad_tk)
        self.transformer = Contextual_Encodings_transformer(transformer)
        self.postprocessor = Contextual_Encodings_postprocessor(sentence_length,context_length,train_time_merge_strategy,test_time_merge_strategy,with_cls_embeddings)
        
    def forward(self,x):
        out = self.preprocessor(x)
        y = self.transformer(out["dataloader"])
        retval = self.postprocessor(y,out["segment_ids"])
        return retval
