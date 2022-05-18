import os
import torch
import numpy as np
from tqdm import tqdm,trange
from eval_script import *
from PredictionModule import EstimateCMEE

def training_loop(cmee,train_data,epochs,criterion,optimizer,scheduler,test_dir_path,system_output_path,MODEL_SAVE_PATH,max_grad_norm = 1.0,early_stopping_criteria = None,patience = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cmee.cuda()
    if early_stopping_criteria is not None:
        curp = 0
        best_score = 0
        
    history = {"train_loss" : [],
               "valid_loss" : []}
    
    for i in range(1,1+epochs):
        print("Epoch {}".format(i))
        
        # ========================================
        #              Training
        # ========================================

        train_loss = []
        cmee.train()
        for _ in trange(len(train_data)):
       
            cmee.zero_grad()
            loss = criterion(*cmee(train_data[_]))

            loss.backward()
            train_loss.append(loss.item())

            torch.nn.utils.clip_grad_norm_(parameters = cmee.parameters(),max_norm = max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        history["train_loss"].append(sum(train_loss)/len(train_loss))    
        print("Train_Loss:",history["train_loss"][-1])
          
        train_data = np.random.permutation(train_data)
        # ========================================
        #              Validation
        # ========================================
        
        EstimateCMEE(test_dir_path,system_output_path)(cmee)
        main(test_dir_path,system_output_path,verbose = False)
        
        if early_stopping_criteria is not None:
            current_score = early_stopping_criteria()
            if best_score < current_score:
                torch.save(cmee.state_dict(), MODEL_SAVE_PATH)
                best_score = current_score
                curp = 0
            else:
                curp += 1
                if curp > patience:
                    break
            print("Stopping Criteria Score : {} , Steps_with_lower_than_current_criteria_score_maxima : {}".format(current_score,curp))
        else:
            torch.save(deepcopy(cmee.state_dict()), MODEL_SAVE_PATH)
    return history