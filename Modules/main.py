
import sys
import torch
from tqdm import tqdm,trange
import numpy as np

import torch

# This will take 10 minutes to install 
#!pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html &> /dev/null
#import torch_scatter

#!pip install compress_fasttext
#import compress_fasttext

#sys.path.append('../input/cmee-n2c2-harvard-modules')
from DataExtractionModule import DOCUMENT_DATA_EXTRACTOR
from CMEE import CMEE_MODEL
from ContextualEmbeddingViaTransformersModule import Contextual_Representations_Model
from PredictionModule import EstimateCMEE
from eval_script import *
from ModelTrainingModule import *
from MiscellaneousModule import *
from WordEmbeddingsModule import WordEmbeddings
from ThresholdingModule import *
from glob import glob

import re

##################################################
#                    args reading
##################################################
parser = argparse.ArgumentParser()

    ## Required parameters
parser.add_argument("--train_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The training data dir. Should contain the .txt and .ann files for the tasks.")
parser.add_argument("--valid_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The training data dir. Should contain the .txt and .ann files for the tasks.")
parser.add_argument("--experiment_no",
                        default='trial',
                        type=str,
                        required=True,
                        help="name for the experiment to save things properly.")




args = parser.parse_args()
train_dir = args.train_dir #"../input/train"
valid_dir = args.valid_dir #"../input/validation-data"
exp_no = args.experiment_no
#######################################################################################################
#                    Contextual Embeddings using BERT with sliding window
#######################################################################################################

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CRM = Contextual_Representations_Model(transformer = base_model,tokenizer = tokenizer,sentence_length = 32,context_length = 80,batch_size=16,cls_tk = 101,sep_tk = 102,pad_tk = 0,train_time_merge_strategy = "mean",test_time_merge_strategy = "mean",with_cls_embeddings = True)

#######################################################################################################
#                    WordEmbeddings FastText (256) + Glove (300) + Pos-Tagging
#######################################################################################################

WE = WordEmbeddings(path_fasttext = "../WordEmbeddings/WordEmbeddings/FastText256d" ,
                    path_glove = "../WordEmbeddings/WordEmbeddings/GloVe_300d_oa_all.wv")

#######################################################################################################
#                                    Shared Model and HEADS 
#######################################################################################################

shared_model = torch.nn.Identity()
head1 = Linear_and_LSTM_Head(768*2 + 603, 7).to(device)
head2 = Linear_and_LSTM_Head(768*2 + 603, 20).to(device)

cmee = CMEE_MODEL(None,shared_model,head1,head2,SimpleLinearThresholding(),contextual_embeddings = CRM,word_embeddings = WE)


#######################################################################################################
#
#                                 TRAINING DATASET GENERATION
#
#######################################################################################################

train_sentence_length = 512
sentence_offset = 256

#######################################################################################################
train_extr = DOCUMENT_DATA_EXTRACTOR(train_dir+'/',name = "train_extr",tagging = "BIO2",default_open_saved_file_if_exists = True,default_save_read_file = True)

train_data = []
for i in glob(train_dir+"/*"):
    if not i.endswith(".txt"):
        continue
    file_no = i.split("/")[-1].split(".")[0]
    out = train_extr.extract(file_no,training = True)
    for i in range(((len(out["sentences"]) - train_sentence_length + sentence_offset - 1) // sentence_offset) + 1):
        train_data.append({j:(out[j][sentence_offset * i : sentence_offset * i + train_sentence_length] if not j.startswith("tag") else out[j]) for j in out})
        train_data[-1]["attribute_labels"] = torch.tensor(train_data[-1]["attribute_labels"],dtype = torch.long).to(device)
        train_data[-1]["labels"] = torch.tensor(train_data[-1]["labels"],dtype = torch.long).to(device)

#######################################################################################################
#
#                                           UpSampling
#
#######################################################################################################

inds_nodisposition = []
inds_disposition = []
inds_undetermined = []
for i in range(len(train_data)):
    inds_disposition.append(torch.any(torch.logical_or(1 == train_data[i]["labels"],2 == train_data[i]["labels"])).detach().cpu().numpy())
    inds_undetermined.append(torch.any(5 <= train_data[i]["labels"]).detach().cpu().numpy())
    inds_nodisposition.append(torch.any(torch.logical_or(3 == train_data[i]["labels"],4 == train_data[i]["labels"])).detach().cpu().numpy())

inds_disposition = np.where(np.stack(inds_disposition,axis = 0))
inds_undetermined = np.where(np.stack(inds_undetermined,axis = 0))
inds_nodisposition = np.where(np.stack(inds_nodisposition,axis = 0))

to_be_added = []
for i in inds_disposition[0]:
    to_be_added.append(train_data[i])

for k in range(4):
    for i in inds_undetermined[0]:
        to_be_added.append(train_data[i])

train_data += to_be_added
#######################################################################################################
#
#                           Optimizer and Scheduler for CMEE-model
#
#######################################################################################################

epochs = 12
max_grad_norm = 1.0

#######################################################################################################

import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

param_optimizer = list(cmee.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

total_steps = len(train_data) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# uncomment to train the model
'''

if not os.path.isdir("./DEV_PREDICTIONS"):
    os.mkdir("./DEV_PREDICTIONS")
if not os.path.isfile("./Saved_model.pt"):
    open("Saved_model.pt","w+")
    
training_loop( cmee , train_data , epochs , LossCMEE(0.333,0.667,wCrossEntropyLoss()) , optimizer , scheduler ,
              valid_dir , "./DEV_PREDICTIONS" , f"./Saved_model_{exp_no}.pt" , 1.0 , lambda : - get_score_fn_for_Task3(valid_dir ,"./DEV_PREDICTIONS")(),20)

'''

cmee.load_state_dict(torch.load(f"./Saved_model_{exp_no}.pt"))

print("trained model loaded")
print("performing final dev prediction")
cmee.contextual_embeddings.postprocessor.merge_strategy = ["max","max"]
# dev predictions
if not os.path.isdir(f"./DEV_PREDICTIONS_{exp_no}"):
    os.mkdir(f"./DEV_PREDICTIONS_{exp_no}")

EstimateCMEE(valid_dir,f"./DEV_PREDICTIONS_{exp_no}")(cmee)
main(valid_dir,f"./DEV_PREDICTIONS_{exp_no}",verbose = False)

# test predictions

print("performing final test prediction")


if not os.path.isdir(f"./TEST_PREDICTIONS_{exp_no}"):
    os.mkdir(f"./TEST_PREDICTIONS_{exp_no}")
# we keep test data same
EstimateCMEE("../input/test-data",f"./TEST_PREDICTIONS_{exp_no}")(cmee)
main("../input/test-data","./TEST_PREDICTIONS_{exp_no}",verbose = False)

