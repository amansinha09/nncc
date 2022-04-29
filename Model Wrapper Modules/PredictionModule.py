from glob import glob
import os
from DataExtractionModule import DOCUMENT_DATA_EXTRACTOR

def get_predictions_fn(out1,out2,data):
    entl = [["NULL",[-1,-1],"NULL",np.array([0]*20)]]
    label_prediction = out1.reshape(-1,7).argmax(axis = -1)
    
    prev_state = 'O'
    for i in range(len(label_prediction)):
        if label_prediction[i] == 1:
            prev_state = 'D'
            entl.append([data["sentences"][i],list(data["spans"][i]),"Disposition",out2[i]])
        elif label_prediction[i] == 3:
            prev_state = 'N'
            entl.append([data["sentences"][i],list(data["spans"][i]),"NoDisposition",out2[i]])
        elif label_prediction[i] == 5:
            prev_state = 'U'
            entl.append([data["sentences"][i],list(data["spans"][i]),"Undetermined",out2[i]])
        elif (prev_state == 'D' and label_prediction[i] == 2) or (prev_state == 'N' and label_prediction[i] == 4) or (prev_state == 'U' and label_prediction[i] == 6):
            entl[-1][0] += " " + data["sentences"][i]
            entl[-1][1][1] = data["spans"][i][1]
            entl[-1][3] += out2[i]
    return entl

class EstimateCMEE():
    
    def __init__(self,test_PATH,save_PATH):
        self.save_path = save_PATH
        self.test_path = test_PATH
        self.extractor = DOCUMENT_DATA_EXTRACTOR(test_PATH+'/',name = "EstimateCMEE",tagging = "BIO2",default_open_saved_file_if_exists = True,default_save_read_file = True)        
        self.pred_fn = get_predictions_fn
        
    def __call__(self,model):
        for i in glob(self.test_path + "/*"):
            if not i.endswith(".txt"):
                continue
            
            file_no = i.split("/")[-1].split('.')[0]
            with torch.no_grad():
                out1,out2,data = model(self.extractor.extract(file_no,training = False))
            
            entl = self.pred_fn(out1,out2,data)
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
                    brat_file.write("A" + str(5*cnt+1) + "\t" + "Action" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][0][item[3][:7].argmax()] + '\n')
                    brat_file.write("A" + str(5*cnt+2) + "\t" + "Temporality" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][1][item[3][7:10].argmax()] + '\n')                    
                    brat_file.write("A" + str(5*cnt+3) + "\t" + "Certainty" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][2][item[3][10:14].argmax()] + '\n')                    
                    brat_file.write("A" + str(5*cnt+4) + "\t" + "Actor" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][3][item[3][14:18].argmax()] + '\n')                    
                    brat_file.write("A" + str(5*cnt+5) + "\t" + "Negation" + " " + "E" + str(index+1) + " " + data["tagging_scheme"]["attr_types_tags"][4][item[3][18:].argmax()] + '\n') 
