import torch
import numpy as np
from glob import glob
import os
from DataExtractionModule import DOCUMENT_DATA_EXTRACTOR

def get_predictions_fn(out1,out2,data,threshold_strategy):
    entl = [["NULL",[-1,-1],"NULL",np.array([0.]*20)]]
    label_prediction = threshold_strategy(out1)
    
    prev_state = 'O'
    cnt = 1.
    for i in range(len(label_prediction)):
        if label_prediction[i] == 1:
            entl[-1][3] /= cnt
            prev_state = 'D'
            entl.append([data["sentences"][i],list(data["spans"][i]),"Disposition",out2[i]])
            cnt = 1
        elif label_prediction[i] == 3:
            entl[-1][3] /= cnt
            prev_state = 'N'
            entl.append([data["sentences"][i],list(data["spans"][i]),"NoDisposition",out2[i]])
            cnt = 1
        elif label_prediction[i] == 5:
            entl[-1][3] /= cnt
            prev_state = 'U'
            entl.append([data["sentences"][i],list(data["spans"][i]),"Undetermined",out2[i]])
            cnt = 1
        elif (prev_state == 'D' and label_prediction[i] == 2) or (prev_state == 'N' and label_prediction[i] == 4) or (prev_state == 'U' and label_prediction[i] == 6):
            entl[-1][0] += " " + data["sentences"][i]
            entl[-1][1][1] = data["spans"][i][1]
            entl[-1][3] += out2[i]
            cnt += 1
        else:
            prev_state = 'O'
    entl[-1][3] /= cnt
    return entl

class EstimateCMEE():
    
    def __init__(self,test_PATH,save_PATH):
        self.save_path = save_PATH
        self.test_path = test_PATH
        self.extractor = DOCUMENT_DATA_EXTRACTOR(test_PATH+'/',name = "EstimateCMEE",tagging = "BIO2",default_open_saved_file_if_exists = True,default_save_read_file = True)        
        self.pred_fn = get_predictions_fn
        
    def __call__(self,model):
        model.eval()
        for i in glob(self.test_path + "/*"):
            if not i.endswith(".txt"):
                continue
            
            file_no = i.split("/")[-1].split('.')[0]
            with torch.no_grad():
                out1,out2,data = model(self.extractor.extract(file_no,training = False))
            
            entl = self.pred_fn(out1,out2,data,model.thresholding)
            data["tagging_scheme"] = {}
            data["tagging_scheme"]["attr_types_tags"] = [['Start', 'Stop', 'Increase', 'Decrease', 'OtherChange', 'UniqueDose', 'Unknown'],
                                                         ['Physician', 'Patient', 'Unknown'],
                                                         ['Certain', 'Hypothetical', 'Conditional', 'Unknown'],
                                                         ['Past', 'Present', 'Future', 'Unknown'],
                                                         ['Negated', 'NotNegated']]

            
            file_path = self.save_path + '/' + file_no + ".ann"
            if os.path.isfile(file_path):
                os.remove(file_path)
            cnt = -1
            with open(file_path,"w+") as brat_file:
                for index,item in enumerate(entl[1:]):
                    brat_file.write("T" + str(index+1) + "\t" + item[2] + ' ' + " ".join([str(k) for k in item[1]]) + "\t" + item[0] + '\n')
                    brat_file.write("E" + str(index+1) + "\t" + item[2] + ":" + "T" + str(index+1) + "\n")
                    if item[2] != "Disposition":
                        continue
                    cnt += 1 
                    brat_file.write("A" + str(5*cnt+1) + "\t" + "Action" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][0][model.thresholding(item[3][:7],index = "Action")[0]] + '\n')
                    brat_file.write("A" + str(5*cnt+2) + "\t" + "Actor" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][1][model.thresholding(item[3][7:10],index = "Actor")[0]] + '\n')                    
                    brat_file.write("A" + str(5*cnt+3) + "\t" + "Certainty" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][2][model.thresholding(item[3][10:14],index = "Certainty")[0]] + '\n')                    
                    brat_file.write("A" + str(5*cnt+4) + "\t" + "Temporality" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][3][model.thresholding(item[3][14:18],index = "Temporality")[0]] + '\n')                    
                    brat_file.write("A" + str(5*cnt+5) + "\t" + "Negation" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][4][model.thresholding(item[3][18:],index = "Negation")[0]] + '\n') 


def get_predictions_fn_2(out1,out2,data,threshold_strategy,provided_data):
    out1 = out1.cpu().numpy()
    out2 = out2.cpu().numpy()
    provided_data = list(reversed(provided_data))
    
    if len(provided_data) == 0:
        return []
    
    entl = [["NULL",provided_data[-1][1][1:],np.array([0.]*3),np.array([0.]*20)]]

    for i in range(len(data["spans"])):
        status = False
        while len(provided_data) and provided_data[-1][0][1] < data["spans"][i][0]:
            provided_data.pop()
            status = True
        if len(provided_data) == 0:
            break

        if status:
            entl.append(["NULL",provided_data[-1][1][1:],np.array([0.]*3),np.array([0.]*20)])

        if provided_data[-1][0][0] <= data["spans"][i][0] and data["spans"][i][1] <= provided_data[-1][0][1]:
            entl[-1][3] += out2[i]
            entl[-1][2] += np.array([out1[i][1] + out1[i][2],out1[i][3] + out1[i][4],out1[i][5] + out1[i][6]])

    for e in entl:
        e[2] = ["Disposition","NoDisposition","Undetermined"][np.argmax(e[2])]

    return entl

class EstimateCMEE2():
    def __init__(self,test_PATH,save_PATH):
        self.save_path = save_PATH
        self.test_path = test_PATH
        self.extractor = DOCUMENT_DATA_EXTRACTOR(test_PATH+'/',name = "EstimateCMEE",tagging = "BIO2",default_open_saved_file_if_exists = True,default_save_read_file = True)        
        self.pred_fn = get_predictions_fn_2

    def __call__(self,model):
        model.eval()
        for i in glob(self.test_path + "/*"):
            if not i.endswith(".txt"):
                continue

            file_no = i.split("/")[-1].split('.')[0]
            with torch.no_grad():
                out1,out2,data = model(self.extractor.extract(file_no,training = False))

            provided_data = []
            
            with open(self.test_path + '/' + file_no + ".ann",'r') as txt:
                txt = txt.read().split('\n')
                for e in txt:
                    if not e.startswith('T'):
                        continue
                    provided_data.append([list(int(x) for x in (e.split('\t')[-2].split(' ')[1],e.split('\t')[-2].split(' ')[-1])),e.split('\t')[0]])

            provided_data = sorted(provided_data)
                
            entl = self.pred_fn(out1,out2,data,model.thresholding,provided_data)
            data["tagging_scheme"] = {}
            data["tagging_scheme"]["attr_types_tags"] = [['Start', 'Stop', 'Increase', 'Decrease', 'OtherChange', 'UniqueDose', 'Unknown'],
                                                         ['Physician', 'Patient', 'Unknown'],
                                                         ['Certain', 'Hypothetical', 'Conditional', 'Unknown'],
                                                         ['Past', 'Present', 'Future', 'Unknown'],
                                                         ['Negated', 'NotNegated']]
                
            file_path = self.save_path + '/' + file_no + ".ann"
            if os.path.isfile(file_path):
                os.remove(file_path)

            cnt = -1

            with open(file_path,"w+") as brat_file:
                with open(self.test_path + '/' + file_no + ".ann",'r') as txt:
                    txt = txt.read().split('\n')
                    for e in txt:
                        if not e.startswith('T'):
                            continue
                        if e.startswith('T'):
                            brat_file.write(re.sub("Disposition|NoDisposition|Undetermined","Drug",e) + '\n')
                for index,item in enumerate(entl):
                    brat_file.write("E" + item[1] + "\t" + item[2] + ":" + "T" + item[1] + "\n")
                    if item[2] != "Disposition":
                        continue
                    cnt += 1 

                    brat_file.write("A" + str(5*cnt+1) + "\t" + "Action" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][0][model.thresholding(item[3][:7],index = "Action")[0]] + '\n')
                    brat_file.write("A" + str(5*cnt+2) + "\t" + "Actor" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][1][model.thresholding(item[3][7:10],index = "Actor")[0]] + '\n')                    
                    brat_file.write("A" + str(5*cnt+3) + "\t" + "Certainty" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][2][model.thresholding(item[3][10:14],index = "Certainty")[0]] + '\n')                    
                    brat_file.write("A" + str(5*cnt+4) + "\t" + "Temporality" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][3][model.thresholding(item[3][14:18],index = "Temporality")[0]] + '\n')                    
                    brat_file.write("A" + str(5*cnt+5) + "\t" + "Negation" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][4][model.thresholding(item[3][18:],index = "Negation")[0]] + '\n') 


def get_predictions_fn_3(out1,out2,data,threshold_strategy,provided_data):
    out1 = out1.cpu().numpy()
    out2 = out2.cpu().numpy()
    provided_data = list(reversed(provided_data))
    
    if len(provided_data) == 0:
        return []
    
    entl = [["NULL",provided_data[-1][-2][1:],provided_data[-1][-1],np.array([0.]*20)]]

    for i in range(len(data["spans"])):
        status = False
        while len(provided_data) and provided_data[-1][0][1] < data["spans"][i][0]:
            provided_data.pop()
            status = True
        if len(provided_data) == 0:
            break

        if status:
            entl.append(["NULL",provided_data[-1][-2][1:],provided_data[-1][-1],np.array([0.]*20)])

        if provided_data[-1][0][0] <= data["spans"][i][0] and data["spans"][i][1] <= provided_data[-1][0][1]:
            entl[-1][3] += out2[i]
            
    return entl

class EstimateCMEE3():
    def __init__(self,test_PATH,save_PATH):
        self.save_path = save_PATH
        self.test_path = test_PATH
        self.extractor = DOCUMENT_DATA_EXTRACTOR(test_PATH+'/',name = "EstimateCMEE",tagging = "BIO2",default_open_saved_file_if_exists = True,default_save_read_file = True)        
        self.pred_fn = get_predictions_fn_3

    def __call__(self,model):
        model.eval()
        for i in glob(self.test_path + "/*"):
            if not i.endswith(".txt"):
                continue

            file_no = i.split("/")[-1].split('.')[0]
            with torch.no_grad():
                out1,out2,data = model(self.extractor.extract(file_no,training = False))

            provided_data = []
            
            event_map = {}
            
            with open(self.test_path + '/' + file_no + ".ann",'r') as txt:
                txt = txt.read().split('\n')
                for e in txt:
                    if not (e.startswith('T') or e.startswith('E')):
                        continue
                    if e.startswith('T'):
                        provided_data.append([list(int(x) for x in (e.split('\t')[-2].split(' ')[1],e.split('\t')[-2].split(' ')[-1])),e.split('\t')[0]])
                    else:
                        event_map[e.split(':')[-1][:-1]] = [e.split("\t")[0],e.split("\t")[1].split(':')[0]]
            
            for e in provided_data:
                e += (event_map[e[-1]])
            
            del event_map
            
            provided_data = sorted(provided_data)
            
            entl = self.pred_fn(out1,out2,data,model.thresholding,provided_data)
            data["tagging_scheme"] = {}
            data["tagging_scheme"]["attr_types_tags"] = [['Start', 'Stop', 'Increase', 'Decrease', 'OtherChange', 'UniqueDose', 'Unknown'],
                                                         ['Physician', 'Patient', 'Unknown'],
                                                         ['Certain', 'Hypothetical', 'Conditional', 'Unknown'],
                                                         ['Past', 'Present', 'Future', 'Unknown'],
                                                         ['Negated', 'NotNegated']]
                
            file_path = self.save_path + '/' + file_no + ".ann"
            if os.path.isfile(file_path):
                os.remove(file_path)

            cnt = -1

            with open(file_path,"w+") as brat_file:
                with open(self.test_path + '/' + file_no + ".ann",'r') as txt:
                    txt = txt.read().split('\n')
                    for e in txt:
                        if not (e.startswith('T') or e.startswith('E')): 
                            continue
                        if e.startswith('T'):
                            brat_file.write(re.sub("Disposition|NoDisposition|Undetermined","Drug",e) + '\n')
                        else:
                            brat_file.write(e + '\n')
                            
                for index,item in enumerate(entl):
                    if item[2] != "Disposition":
                        continue
                    cnt += 1 
                    
                    brat_file.write("A" + str(5*cnt+1) + "\t" + "Action" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][0][model.thresholding(item[3][:7],index = "Action")[0]] + '\n')
                    brat_file.write("A" + str(5*cnt+2) + "\t" + "Actor" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][1][model.thresholding(item[3][7:10],index = "Actor")[0]] + '\n')                    
                    brat_file.write("A" + str(5*cnt+3) + "\t" + "Certainty" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][2][model.thresholding(item[3][10:14],index = "Certainty")[0]] + '\n')                    
                    brat_file.write("A" + str(5*cnt+4) + "\t" + "Temporality" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][3][model.thresholding(item[3][14:18],index = "Temporality")[0]] + '\n')                    
                    brat_file.write("A" + str(5*cnt+5) + "\t" + "Negation" + " " + "E" + item[1] + " " + data["tagging_scheme"]["attr_types_tags"][4][model.thresholding(item[3][18:],index = "Negation")[0]] + '\n')