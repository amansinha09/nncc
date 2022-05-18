
#  python PATH_TO_training.py --train_dir PATH_TO_TRAIN_FOLDER --valid_dir  PATH_TO_DEV_FOLDER --output_folder  PATH_TO_SYSTEM_OUTPUT_FOLDER --save_path PATH_TO_save_model.pt --head linear|linear_lstm --on_task task_1|task_2

import torch
import torch_scatter
import compress_fasttext
import numpy as np
from tqdm import tqdm,trange
import os

import argparse

import sys
sys.path.append(os.path.dirname(__file__) + "/Modules")
from eval_script import *
from DataExtractionModule import DOCUMENT_DATA_EXTRACTOR
from CMEE import CMEE_MODEL
from ContextualEmbeddingViaTransformersModule import Contextual_Representations_Model
from PredictionModule import EstimateCMEE
from ModelTrainingModule import *
from MiscellaneousModule import *
from WordEmbeddingsModule import WordEmbeddings
from ThresholdingModule import *
from glob import glob

if __name__ == '__main__':
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
    parser.add_argument("--output_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="The folder where all system output results will be saved.")
    parser.add_argument("--save_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The path to which model is to be saved.Original contents of this file will be overridden")
    parser.add_argument("--head",
                        default=None,
                        type=str,
                        required=True,
                        help="Classification Head: 'linear' , 'linear_lstm' ")
    parser.add_argument("--on_task",
                        default=None,
                        type=str,
                        required=True,
                        help="The task to be optimized.Valid values: 'task_1' , 'task_2' ")
    args = parser.parse_args()
    train_dir = args.train_dir
    valid_dir = args.valid_dir
    head = args.head
    on_task = args.on_task
    output_folder = args.output_folder
    save_path = args.save_path
    #######################################################################################################
    #                    Contextual Embeddings using BERT with sliding window
    #######################################################################################################

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    CRM = Contextual_Representations_Model(transformer = base_model,tokenizer = tokenizer,sentence_length = 32,context_length = 80,batch_size=16,cls_tk = 101,sep_tk = 102,pad_tk = 0,train_time_merge_strategy = "mean",test_time_merge_strategy = "mean",with_cls_embeddings = True)

    #######################################################################################################
    #                    WordEmbeddings FastText (256) + Glove (300) + Pos-Tagging
    #######################################################################################################

    WE = WordEmbeddings(path_fasttext = os.path.dirname(__file__) + "/Pretrained Word Embeddings/FastText256d" ,
                        path_glove = os.path.dirname(__file__) + "/Pretrained Word Embeddings/GloVe_300d_oa_all.wv")

    #######################################################################################################
    #                                    Shared Model and HEADS 
    #######################################################################################################

    shared_model = torch.nn.Identity()

    if args.head == "linear":
        head1 = Linear_Head(768*2 + 603 , 7).cuda()
        head2 = Linear_Head(768*2 + 603 , 20).cuda()
    else :
        head1 = Linear_and_LSTM_Head(768*2 + 603 , 7).cuda()
        head2 = Linear_and_LSTM_Head(768*2 + 603 , 20).cuda()

    #######################################################################################################

    cmee = CMEE_MODEL(None,shared_model,head1,head2,SimpleLinearThresholding(),contextual_embeddings = CRM,word_embeddings = WE)

    #######################################################################################################
    #
    #                                 TRAINING DATASET GENERATION
    #
    #######################################################################################################

    train_sentence_length = 512
    sentence_offset = 256

    #######################################################################################################
    train_extr = DOCUMENT_DATA_EXTRACTOR(train_dir + '/',name = "train_extr",tagging = "BIO2",default_open_saved_file_if_exists = True,default_save_read_file = True)

    train_data = []
    for i in glob(train_dir + "/*"):
        if not i.endswith(".txt"):
            continue
        file_no = i.split("/")[-1].split(".")[0]
        out = train_extr.extract(file_no,training = True)
        for i in range(((len(out["sentences"]) - train_sentence_length + sentence_offset - 1) // sentence_offset) + 1):
            train_data.append({j:(out[j][sentence_offset * i : sentence_offset * i + train_sentence_length] if not j.startswith("tag") else out[j]) for j in out})
            train_data[-1]["attribute_labels"] = torch.tensor(train_data[-1]["attribute_labels"],dtype = torch.long).to("cuda")
            train_data[-1]["labels"] = torch.tensor(train_data[-1]["labels"],dtype = torch.long).to("cuda")

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

    for i in inds_disposition[0]:
        train_data.append(train_data[i])

    for k in range(4):
        for i in inds_undetermined[0]:
            train_data.append(train_data[i])

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

    if args.head == "linear":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else :
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and "bidirectional_LSTM" not in n],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and "bidirectional_LSTM" not in n],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in param_optimizer if "bidirectional_LSTM" in n],
             'weight_decay_rate': 0.01,"lr": 1e-3}
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

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    if not os.path.isfile(save_path):
        open(save_path,"w+")

    if args.on_task == "task_2":
        training_loop( cmee , train_data , epochs , LossCMEE(1.,0.,wCrossEntropyLoss()) , optimizer , scheduler ,
                      args.valid_dir , output_folder , save_path , 1.0 , lambda : - get_score_fn_for_Task2(valid_dir ,output_folder)())
        cmee.load_state_dict(torch.load(save_path))    
        study = cmee.thresholding.optimize_parameters(cmee,get_score_fn_for_Task2,gold_standard_folder = valid_dir,system_output_folder = output_folder,n_trials = 128,verbose = False)
        print("Final Results after adjusting threshold")
        EstimateCMEE(valid_dir,output_folder)(cmee)
        main(valid_dir,output_folder,verbose = False)
    else:
        training_loop( cmee , train_data , epochs , LossCMEE(1.,5.,wCrossEntropyLoss()) , optimizer , scheduler ,
                      valid_dir , output_folder , save_path , 1.0 , lambda : - get_score_fn_for_Task3(valid_dir ,output_folder)())
        cmee.load_state_dict(torch.load(save_path))        
        study = cmee.thresholding.optimize_parameters(cmee,get_score_fn_for_Task3,gold_standard_folder = valid_dir,system_output_folder = output_folder,n_trials = 128,verbose = False)
        print("Final Results after adjusting threshold")
        EstimateCMEE(valid_dir,output_folder)(cmee)
        main(valid_dir,output_folder,verbose = False)