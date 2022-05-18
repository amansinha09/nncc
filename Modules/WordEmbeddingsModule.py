import torch
import numpy as np
import gensim
import compress_fasttext
import nltk

class WordEmbeddings():
    def __init__(self,path_fasttext,path_glove):
        self.fasttext = compress_fasttext.models.CompressedFastTextKeyedVectors.load(path_fasttext)
        self.glove = gensim.models.KeyedVectors.load_word2vec_format(path_glove,binary = True)
        self.glove.add_vector(key = "UNK",vector = self.glove.vectors.mean(axis = 0))
        self.postag_to_index = {i:j for j,i in enumerate(['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS', 'UNK'])}
        self.one_hot_vectors = np.eye(len(self.postag_to_index))
        self.get_postag_vector_fn = lambda x: self.one_hot_vectors[self.postag_to_index[x if x in self.postag_to_index else "UNK"]]
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    def __call__(self,x):
        l = []
        for i in x:
            i = i.lower()
            l.append(np.concatenate([self.fasttext[i],self.glove[(i if self.glove.has_index_for(i) else "UNK")],np.array([1 if self.glove.has_index_for(i) else 0]),self.get_postag_vector_fn(nltk.pos_tag([i])[0][1])],axis = 0))
        return torch.tensor(np.array(l),dtype = torch.float32).to(self.device)
    
    def to(self,*args):
        return self