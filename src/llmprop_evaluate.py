import os
import re
import glob
import time
import datetime
import random
from datetime import timedelta

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import SGD

import matplotlib.pyplot as plt

# add the progress bar
from tqdm import tqdm

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer, BertTokenizerFast, BertModel
from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()

# pre-defined functions
from llmprop_model import Predictor
from llmprop_utils import *
from llmprop_dataset import *
from llmprop_args_parser import *

# for metrics
from torchmetrics.classification import BinaryAUROC
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr, kendalltau
from statistics import stdev

# for Weight&Biases
import wandb
from wandb import AlertLevel
from datetime import timedelta

# import bitsandbytes as bnb
# from bitsandbytes.optim import Adam8bit
import subprocess

def evaluate(
    model, 
    mae_loss_function, 
    test_dataloader, 
    train_labels_mean, 
    train_labels_std, 
    property_name,
    device,
    task_name,
    normalizer="z_norm"
):
    test_start_time = time.time()

    model.eval()

    total_test_loss = 0
    predictions_list = []
    targets_list = []
    embeddings_list = []

    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            if preprocessing_strategy == 'xVal':
                batch_inputs, batch_masks, batch_labels, batch_x_num = tuple(b.to(device) for b in batch)
                _, predictions = model(batch_inputs, batch_masks, x_num=batch_x_num)
            else:
                batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
                _, predictions = model(batch_inputs, batch_masks)

            if task_name == "classification":
                predictions_denorm = predictions

            elif task_name == "regression":
                if normalizer == 'z_norm':
                    predictions_denorm = z_denormalize(predictions, train_labels_mean, train_labels_std)

                elif normalizer == 'mm_norm':
                    predictions_denorm = mm_denormalize(predictions, train_labels_min, train_labels_max)

                elif normalizer == 'ls_norm':
                    predictions_denorm = ls_denormalize(predictions)

                elif normalizer == 'no_norm':
                    predictions_denorm = predictions

        predictions_detached = predictions_denorm.detach().cpu().numpy()
        targets = batch_labels.detach().cpu().numpy()
        # embeddings_detached = embeddings.detach().cpu().numpy()

        for i in range(len(predictions_detached)):
            predictions_list.append(predictions_detached[i][0])
            targets_list.append(targets[i])
            # embeddings_list.append(embeddings_detached[i])
        
    if task_name == "classification":
        test_performance = get_roc_score(predictions_list, targets_list)
        print(f"\n Test ROC score on predicting {property_name} = {test_performance}")

    elif task_name == "regression":
        predictions_tensor = torch.tensor(predictions_list)
        targets_tensor = torch.tensor(targets_list)
        test_performance = mae_loss_function(predictions_tensor.squeeze(), targets_tensor.squeeze())
        r2 = metrics.r2_score(targets_list, predictions_list)

        print(f"\n The test performance on predicting {property}:")
        print(f"MAE error = {test_performance}")
        print(f"R2 score = {r2}")

    average_test_loss = total_test_loss / len(test_dataloader)
    test_ending_time = time.time()
    testing_time = time_format(test_ending_time-test_start_time)
    print(f"Testing took {testing_time} \n")

    return predictions_list, test_performance

if __name__ == "__main__":
    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
        n_gpus = torch.cuda.device_count()
        print("Evaluating on", n_gpus, "GPUs!")
        print('-'*50)
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")
        
    # parse Arguments
    args = args_parser()
    config = vars(args)
    
    inference_batch_size = config.get('inference_bs')
    max_length = config.get('max_len')
    drop_rate = config.get('dr')
    preprocessing_strategy = config.get('preprocessing_strategy')
    tokenizer_name = config.get('tokenizer')
    pooling = config.get('pooling')
    normalizer_type = config.get('normalizer')
    property_name = config.get('property_name')
    data_path = config.get('data_path')
    input_type = config.get('input_type')
    dataset_name = config.get('dataset_name')
    model_name = config.get('model_name')
    task_name = "regression"
    iteration_no = 1
    additional_samples_type = 'top_10k'
    
    if model_name == "matbert":
        pooling = None

    # prepare the data 
    def concatenate_and_shuffle(df_1, df_2):
        concatenated_df = pd.concat([df_1, df_2], ignore_index=True)
        shuffled_df = concatenated_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return shuffled_df

    test_data = pd.read_csv(f"data/{property_name.lower()}/mofseq/test.csv")
    
    # drop duplicates in test data
    if input_type in ["mof_name","mofkey","mofid_v1"]:
        test_data = test_data.dropna(subset=[input_type]).reset_index(drop=True)

    # define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # add defined special tokens to the tokenizer
    if pooling == 'cls':
        tokenizer.add_tokens(["[CLS]"])
    
    if preprocessing_strategy == "xVal":
        tokenizer.add_tokens(["[NUM]"])
        
    if input_type == "mof_name_and_cif_string":
        tokenizer.add_tokens(["[SEP]"])
    elif input_type == "combined_mof_str":
        tokenizer.add_tokens(["<mofname>","</mofname>",
                              "<mofid>","</mofid>",
                              "<mofkey>","</mofkey>"
                              ])
    elif input_type == "mofseq":
        tokenizer.add_tokens(["<mofname>","</mofname>",
                              "<mofid>","</mofid>",
                              ])
    
    if input_type == "mofkey":
        test_data[input_type] = test_data[input_type].apply(clean_mofkey)
    elif input_type == "mofid_v1":
        test_data[input_type] = test_data[input_type].apply(clean_mofid)
    elif input_type == "combined_mof_str":
        test_data = combine_mof_string_representations(test_data, tokenizer, input_type, max_length=max_length) 
    elif input_type == "mofseq":
        test_data = generate_mofseq(test_data, tokenizer, input_type, max_length=max_length)
        
    test_data = test_data.drop_duplicates(subset=[input_type]).reset_index(drop=True)
    
    # process train data labels for denormalization
    train_data_config = readJSON(f"checkpoints/{property_name.lower()}/mofseq_config.json")
    train_labels_mean = torch.tensor(train_data_config['train_data_info'][f'mean_{property_name}'], dtype=torch.float32)
    train_labels_std = torch.tensor(train_data_config['train_data_info'][f'std_{property_name}'], dtype=torch.float32)
    train_labels_min = torch.tensor(train_data_config['train_data_info'][f'min_{property_name}'], dtype=torch.float32)
    train_labels_max = torch.tensor(train_data_config['train_data_info'][f'max_{property_name}'], dtype=torch.float32) 

    if preprocessing_strategy == "none":
        test_data = test_data

    # define loss functions
    mae_loss_function = nn.L1Loss()
    bce_loss_function = nn.BCEWithLogitsLoss()

    freeze = False # a boolean variable to determine if we freeze the pre-trained T5 weights 
    
    data_to_name = {
        '0':'train',
        '1':'test',
        '2':'valid'
    }
    input_to_ckpt = {
        'mof_name': '/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/checkpoints/1m_mof/mofbench_llmprop_best_checkpoint_for_FE_atom_regression_mof_name_none_153_tokens_300_epochs_0.001_0.2_100.0%_no_outliers.pt',
        'mofkey':'/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/checkpoints/1m_mof/mofbench_llmprop_best_checkpoint_for_FE_atom_regression_mofkey_none_102_tokens_300_epochs_0.001_0.2_100.0%_no_outliers.pt', 
        'mofid_v1':'/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/checkpoints/1m_mof/mofbench_llmprop_best_checkpoint_for_FE_atom_regression_mofid_v1_none_2000_tokens_200_epochs_0.001_0.2_100.0%_no_outliers.pt',
        'mofname_and_mofid':f'/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/checkpoints/1m_mof/mofbench_llmprop_finetune_iteration_{iteration_no}_{additional_samples_type}_best_checkpoint_for_FE_atom_regression_mofname_and_mofid_none_2000_tokens_200_epochs_0.001_0.2_100.0%_no_outliers.pt',
    }
    
    #get the length of the longest composition
    if input_type in ["mof_name","mofkey"]:
        max_length = get_max_len(test_data, tokenizer, input_type)
        print('\nThe longest composition has', max_length, 'tokens\n')

    print('max length:', max_length)
    if torch.cuda.is_available():
        if max_length <= 888:
            inference_batch_size = 128 * n_gpus
        elif max_length > 888 and max_length <= 1500:
            inference_batch_size = 64 * n_gpus
        elif max_length > 1500 and max_length <= 2000:
            inference_batch_size = 32 * n_gpus

    print("labels statistics on training set:")
    print("Mean:", train_labels_mean)
    print("Standard deviation:", train_labels_std)
    print("Max:", train_labels_max)
    print("Min:", train_labels_min)
    print("-"*50)
    
    print("======= Evaluating on test set ========")
    
    # averaging the results over 5 runs
    predictions = []
    test_results = []
    seed = 42
    offset = 10
    
    for i in range(1):
        np.random.seed(seed + (i*offset))
        random.seed(seed + (i*offset))
        torch.manual_seed(seed + (i*offset))
        torch.cuda.manual_seed(seed + (i*offset))
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed + (i*offset))

        # define the model
        if model_name in ["llmprop", "llmprop_finetune"]:
            base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small") 
            base_model_output_size = 512
        
        # freeze the pre-trained LM's parameters
        if freeze:
            for param in base_model.parameters():
                param.requires_grad = False

        # resizing the model input embeddings matrix to adapt to newly added tokens by the new tokenizer
        # this is to avoid the "RuntimeError: CUDA error: device-side assert triggered" error
        base_model.resize_token_embeddings(len(tokenizer))
        
        # best_model_path = input_to_ckpt[input_type]
        best_model_path = f'checkpoints/{property_name.lower()}/best_llmprop-mofseq_checkpoint_for_MOF_{property_name}_prediction.pt'
        best_model = Predictor(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling, model_name=model_name)

        device_ids = [d for d in range(torch.cuda.device_count())]

        if torch.cuda.is_available():
            best_model = nn.DataParallel(best_model, device_ids=device_ids).cuda()
        else:
            best_model.to(device)

        if isinstance(best_model, nn.DataParallel):
            best_model.module.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False)
        else:
            best_model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False) 
            best_model.to(device)
        
        dataloader = create_dataloaders(
            tokenizer, 
            test_data, 
            max_length, 
            inference_batch_size, 
            property_value=property_name, 
            pooling=pooling,
            normalize=False,
            input_type=input_type,
            preprocessing_strategy=preprocessing_strategy
        )
        
        predictions_list, test_performance = evaluate(best_model, mae_loss_function, dataloader, train_labels_mean, train_labels_std, property, device, task_name, normalizer=normalizer_type)
        predictions.append(predictions_list)
        test_results.append(test_performance)

        # save the averaged predictions
        data['predicted_FE_atom'] = predictions_list
        data.to_csv(f"results/{property_name.lower()}/llmprop_finetune-mofseq_iteration_test_stats_for_{property_name}_regression_none_2000_tokens_200_epochs_0.001_0.2_100.0%_no_outliers.csv")
        
        test_predictions = {f"mof_name":list(test_data['mof_name']), f"actual_{property}":list(test_data[property]), f"predicted_{property}":averaged_predictions}
        saveCSV(pd.DataFrame(test_predictions), f"{statistics_directory}/llm4mat_rebuttal_{model_name}_test_stats_for_{property}_{task_name}_{input_type}_{preprocessing_strategy}_{max_length}_tokens_200_epochs.csv")
        