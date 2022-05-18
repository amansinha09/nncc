from scipy.special import softmax
import optuna
import torch
import numpy as np
from eval_script import *
from PredictionModule import EstimateCMEE

class NoThresholding():
    def __init__(self):
        super().__init__()
        self.n_classes = {0:7,"Action":7,"Actor":3,"Certainty":4,"Temporality":4,"Negation":2}
        
    def __call__(self,x,index = 0):
        return x.reshape(-1,self.n_classes[index]).argmax(axis = -1)
    
    def optimize_parameters(self,**args):
        pass

class SimpleLinearThresholding(): # Simple Linear Threshold on Probability Space
    def __init__(self):
        super().__init__()        
        self.n_classes = {0:7,"Action":7,"Actor":3,"Certainty":4,"Temporality":4,"Negation":2}
        self.parameters = {0:softmax(np.zeros(7)),"Action":softmax(np.zeros(7)),"Actor":softmax(np.zeros(3)),"Certainty":softmax(np.zeros(4)),"Temporality":softmax(np.zeros(4)),"Negation":softmax(np.zeros(2))}
    
    def __call__(self,x,index = 0):
        if type(x) == torch.Tensor:
            if x.device.type == "cuda":
                x = x.detach().cpu()
            x = x.numpy()
            
        return np.argmax(softmax(x.reshape(-1,self.n_classes[index]),axis = -1) - self.parameters[index],axis = -1)
    
    def objective_simple_threshold(self,trial):
        params = np.array([- np.log(trial.suggest_float(f"x_{i}", 0, 1)) for i in range(27)], np.float32)
        n_params = self.convert_scores_to_probs(params)
        for i in range(27):
            trial.set_user_attr(f"p_{i}", n_params[i])
        params = n_params
        params = {0:params[:7],"Action":params[7:14],"Actor":params[14:17],"Certainty":params[17:21],"Temporality":params[21:25],"Negation":params[25:]}
        self.parameters = params
        self.estimator(self.model)
        return self.compute_score()
    
    def optimize_parameters(self,model,score_fn,gold_standard_folder,system_output_folder,study = None,n_trials = 60,verbose = True):
        if study == None:
            study = optuna.create_study()
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        self.model = model
        self.estimator = EstimateCMEE(gold_standard_folder,system_output_folder)
        self.compute_score = score_fn(gold_standard_folder,system_output_folder)
        study.optimize(self.objective_simple_threshold,n_trials = n_trials)
        params = np.array([- np.log(study.best_params["x_" + str(i)]) for i in range(27)],np.float32)
        params = self.convert_scores_to_probs(params)
        self.parameters = {0:params[:7],"Action":params[7:14],"Actor":params[14:17],"Certainty":params[17:21],"Temporality":params[21:25],"Negation":params[25:]}
        self.estimator(self.model)
        print(-self.compute_score())
        del self.model,self.estimator,self.compute_score
        return study
    
    @staticmethod
    def convert_scores_to_probs(params):
        n_params = []
        for i in range(0,7):
            n_params.append(params[i] / sum(params[0:7]))
        for i in range(7,14):
            n_params.append(params[i] / sum(params[7:14]))
        for i in range(14,17):
            n_params.append(params[i] / sum(params[14:17]))
        for i in range(17,21):
            n_params.append(params[i] / sum(params[17:21]))
        for i in range(21,25):
            n_params.append(params[i] / sum(params[21:25]))
        for i in range(25,27):
            n_params.append(params[i] / sum(params[25:27]))
        return n_params