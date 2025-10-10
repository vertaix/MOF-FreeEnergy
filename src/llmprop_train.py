"""
Set up the training code 
"""
import os
import re
import glob
import time
import datetime
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
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()

# pre-defined functions
from llmprop_model import Predictor
from llmprop_utils import *
from llmprop_dataset import *
from llmprop_args_parser import *

# for metrics
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr, kendalltau
from statistics import stdev
from sklearn.model_selection import train_test_split

from datetime import timedelta

import subprocess

# set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train(
    model, 
    optimizer, 
    scheduler, 
    bce_loss_function, 
    mae_loss_function,
    epochs, 
    train_dataloader, 
    valid_dataloader, 
    device,  
    normalizer="z_norm",
    # encoding_type=None
):
    
    training_starting_time = time.time()
    training_stats = []
    validation_predictions = {}
    
    best_loss = 1e10
    best_roc = 0.0

    for epoch in range(epochs):
        print(f"========== Epoch {epoch+1}/{epochs} =========")

        epoch_starting_time = time.time() 

        total_training_loss = 0
        total_training_mae_loss = 0
        total_training_normalized_mae_loss = 0

        model.train()

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch_inputs, batch_masks, batch_labels, batch_norm_labels = tuple(b.to(device) for b in batch)
            _, predictions = model(batch_inputs, batch_masks)

            loss = mae_loss_function(predictions.squeeze(), batch_norm_labels.squeeze())
            
            if normalizer == 'z_norm':
                predictions_denorm = z_denormalize(predictions, train_labels_mean, train_labels_std)

            elif normalizer == 'mm_norm':
                predictions_denorm = mm_denormalize(predictions, train_labels_min, train_labels_max)

            elif normalizer == 'ls_norm':
                predictions_denorm = ls_denormalize(predictions)

            elif normalizer == 'no_norm':
                loss = mae_loss_function(predictions.squeeze(), batch_labels.squeeze())
                predictions_denorm = predictions

            mae_loss = mae_loss_function(predictions_denorm.squeeze(), batch_labels.squeeze()) 

            # total training loss on actual output
            total_training_loss += mae_loss.item()

            # back propagate
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # average training loss on actual output
        average_training_loss = total_training_loss/len(train_dataloader) 
        
        epoch_ending_time = time.time()
        training_time = time_format(epoch_ending_time - epoch_starting_time)

        print(f"Average training loss = {average_training_loss}")
        print(f"Training for this epoch took {training_time}")

        # Validation
        print("")
        print("Running Validation ....")

        valid_start_time = time.time()

        model.eval()

        total_eval_mae_loss = 0
        predictions_list = []
        targets_list = []

        for step, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            with torch.no_grad():
                batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
                _, predictions = model(batch_inputs, batch_masks)

                if normalizer == 'z_norm':
                    predictions_denorm = z_denormalize(predictions, train_labels_mean, train_labels_std)

                elif normalizer == 'mm_norm':
                    predictions_denorm = mm_denormalize(predictions, train_labels_min, train_labels_max)

                elif normalizer == 'ls_norm':
                    predictions_denorm = ls_denormalize(predictions)

                elif normalizer == 'no_norm':
                    predictions_denorm = predictions

            predictions = predictions_denorm.detach().cpu().numpy()
            targets = batch_labels.detach().cpu().numpy()

            for i in range(len(predictions)):
                predictions_list.append(predictions[i][0])
                targets_list.append(targets[i])
        
        valid_ending_time = time.time()
        validation_time = time_format(valid_ending_time-valid_start_time)

        # save model checkpoint and the statistics of the epoch where the model performs the best
        predictions_tensor = torch.tensor(predictions_list)
        targets_tensor = torch.tensor(targets_list)
        valid_performance = mae_loss_function(predictions_tensor.squeeze(), targets_tensor.squeeze())
    
        if valid_performance <= best_loss:
            best_loss = valid_performance
            best_epoch = epoch+1

            # save the best model checkpoint
            save_to_path = checkpoints_directory + f"best_{model_name}-{input_type}_checkpoint_for_MOF_{property_name}_prediction_{task_name}_{preprocessing_strategy}_{max_length}_tokens_{epochs}_epochs_{learning_rate}_{drop_rate}_training_size_{int(training_size*100)}%.pt"
            
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_to_path)
            else:
                torch.save(model.state_dict(), save_to_path)
            
            # save statistics of the best model
            training_stats.append(
                {
                    "best_epoch": epoch + 1,
                    "training mae loss": average_training_loss,
                    "validation mae loss": valid_performance,
                    "training time": training_time,
                    "validation time": validation_time
                }
            )

            validation_predictions.update(
                {
                    f"epoch_{epoch+1}": predictions_list
                }
            )

            saveCSV(pd.DataFrame(data=training_stats), f"{statistics_directory}/{model_name}-{input_type}_training_stats_for_{property_name}_{task_name}_{preprocessing_strategy}_{max_length}_tokens_{epochs}_epochs_{learning_rate}_{drop_rate}_training_size_{int(training_size*100)}%.csv")
            saveCSV(pd.DataFrame(validation_predictions), f"{statistics_directory}/{model_name}-{input_type}_validation_stats_for_{property_name}_{task_name}_{preprocessing_strategy}_{max_length}_tokens_{epochs}_epochs_{learning_rate}_{drop_rate}_training_size_{int(training_size*100)}%.csv")
            
        else:
            best_loss = best_loss
        
        print(f"Validation mae error = {valid_performance}")
        print(f"validation took {validation_time}")

    train_ending_time = time.time()
    total_training_time = train_ending_time-training_starting_time

    print("\n========== Training complete ========")
    print(f"Training LLM_Prop on {property_name} prediction took {time_format(total_training_time)}")
    print(f"The lowest validation MAE error on predicting {property_name} = {best_loss} at {best_epoch}th epoch \n")
    
    return training_stats, validation_predictions

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

    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            _, predictions = model(batch_inputs, batch_masks)

            if normalizer == 'z_norm':
                predictions_denorm = z_denormalize(predictions, train_labels_mean, train_labels_std)

            elif normalizer == 'mm_norm':
                predictions_denorm = mm_denormalize(predictions, train_labels_min, train_labels_max)

            elif normalizer == 'ls_norm':
                predictions_denorm = ls_denormalize(predictions)

            elif normalizer == 'no_norm':
                predictions_denorm = predictions

        predictions = predictions_denorm.detach().cpu().numpy()
        targets = batch_labels.detach().cpu().numpy()

        for i in range(len(predictions)):
            predictions_list.append(predictions[i][0])
            targets_list.append(targets[i])

    predictions_tensor = torch.tensor(predictions_list)
    targets_tensor = torch.tensor(targets_list)
    test_performance = mae_loss_function(predictions_tensor.squeeze(), targets_tensor.squeeze())
    r2 = metrics.r2_score(targets_list, predictions_list)

    print(f"\n The test performance on predicting {property_name}:")
    print(f"MAE error = {test_performance}")
    print(f"R2 score = {r2}")

    average_test_loss = total_test_loss / len(test_dataloader)
    test_ending_time = time.time()
    testing_time = time_format(test_ending_time-test_start_time)
    print(f"Testing took {testing_time} \n")

    return predictions_list, test_performance

def concatenate_and_shuffle(df_1, df_2):
    concatenated_df = pd.concat([df_1, df_2], ignore_index=True)
    shuffled_df = concatenated_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return shuffled_df

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # parse Arguments
    args = args_parser()
    config = vars(args)

    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
        print("Training and testing on", torch.cuda.device_count(), "GPUs!")
        print('-'*50)
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")

    # set parameters
    train_batch_size = config.get('train_bs')
    inference_batch_size = config.get('inference_bs')
    max_length = config.get('max_len')
    learning_rate = config.get('lr')
    drop_rate = config.get('dr')
    epochs = config.get('epochs')
    warmup_steps = config.get('warmup_steps')
    preprocessing_strategy = config.get('preprocessing_strategy')
    tokenizer_name = config.get('tokenizer')
    pooling = config.get('pooling')
    scheduler_type = config.get('scheduler')
    normalizer_type = config.get('normalizer')
    property_name = config.get('property_name')
    optimizer_type = config.get('optimizer')
    task_name = config.get('task_name')
    data_path = config.get('data_path')
    input_type = config.get('input_type')
    dataset_name = config.get('dataset_name')
    model_name = config.get('model_name')
    training_size = config.get('training_size')
    finetuned_property_name = "SE_atom"
    iteration_no = config.get('iteration_no')
    additional_samples_type = config.get('additional_samples_type')
    n_gpus = torch.cuda.device_count()

    # checkpoints directory
    checkpoints_directory = f"checkpoints/{property_name.lower()}/"
    if not os.path.exists(checkpoints_directory):
        os.makedirs(checkpoints_directory)

    # training statistics directory
    statistics_directory = f"results/{property_name.lower()}/"
    if not os.path.exists(statistics_directory):
        os.makedirs(statistics_directory)

    train_data = pd.read_csv(f"data/{property_name.lower()}/mofseq/train.csv")
    valid_data = pd.read_csv(f"data/{property_name.lower()}/mofseq/validation.csv")
    test_data = pd.read_csv(f"data/{property_name.lower()}/mofseq/test.csv")
    
    # define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small") 

    # add defined special tokens to the tokenizer
    if pooling == 'cls':
        tokenizer.add_tokens(["[CLS]"]) 
    
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

    #get the length of the longest composition
    if input_type in ["mof_name","mofkey"]:
        max_length = get_max_len(pd.concat([train_data, valid_data, test_data]), tokenizer, input_type)
        print('\nThe longest composition has', max_length, 'tokens\n')

    print('max length:', max_length)
    
    if input_type == "mof_name_and_cif_string":
        train_data[input_type] = train_data['mof_name'] + '[SEP]' + train_data['cif_string']
        valid_data[input_type] = valid_data['mof_name'] + '[SEP]' + valid_data['cif_string']
        test_data[input_type] = test_data['mof_name'] + '[SEP]' + test_data['cif_string']
    elif input_type == "mofkey":
        train_data[input_type] = train_data[input_type].apply(clean_mofkey)
        valid_data[input_type] = valid_data[input_type].apply(clean_mofkey)
        test_data[input_type] = test_data[input_type].apply(clean_mofkey)
    elif input_type == "mofid_v1":
        train_data[input_type] = train_data[input_type].apply(clean_mofid)
        valid_data[input_type] = valid_data[input_type].apply(clean_mofid)
        test_data[input_type] = test_data[input_type].apply(clean_mofid)
    elif input_type == "combined_mof_str":
        train_data = combine_mof_string_representations(train_data, tokenizer, input_type, max_length=max_length)
        valid_data = combine_mof_string_representations(valid_data, tokenizer, input_type, max_length=max_length)
        test_data = combine_mof_string_representations(test_data, tokenizer, input_type, max_length=max_length)
    elif input_type == "mofseq":
        train_data = generate_mofseq(train_data, tokenizer, input_type, max_length=max_length)
        valid_data = generate_mofseq(valid_data, tokenizer, input_type, max_length=max_length)
        test_data = generate_mofseq(test_data, tokenizer, input_type, max_length=max_length)
        
    train_data = train_data.drop_duplicates(subset=[input_type]).reset_index(drop=True)
    valid_data = valid_data.drop_duplicates(subset=[input_type]).reset_index(drop=True)
    test_data = test_data.drop_duplicates(subset=[input_type]).reset_index(drop=True)

    # Reset indices
    print('Resetting indices...')
    start = time.time()
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    print('Resetting complete! Used', time_format(time.time()-start))

    print(f'Final Dataset Sizes - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}')
    print('-'*100)

    train_data = train_data.loc[0:int(len(train_data)*training_size)]
    task_name = 'regression'
    
    train_labels_array = np.array(train_data[property_name])
    train_labels_mean = torch.mean(torch.tensor(train_labels_array))
    train_labels_std = torch.std(torch.tensor(train_labels_array))
    train_labels_min = torch.min(torch.tensor(train_labels_array))
    train_labels_max = torch.max(torch.tensor(train_labels_array))

    if preprocessing_strategy == "none":
        train_data = train_data
        valid_data = valid_data
        test_data = test_data
        print('\n', train_data[input_type][0])
        print('-'*50)

    # define loss functions
    mae_loss_function = nn.L1Loss()
    bce_loss_function = nn.BCEWithLogitsLoss()

    freeze = False # a boolean variable to determine if we freeze the pre-trained T5 weights
    if torch.cuda.is_available():
        if max_length <= 888:
            train_batch_size = 64 * n_gpus
            inference_batch_size = 128 * n_gpus
        elif max_length > 888 and max_length <= 1500:
            train_batch_size = 32 * n_gpus
            inference_batch_size = 64 * n_gpus
        elif max_length > 1500 and max_length <= 2000:
            train_batch_size = 16 * n_gpus
            inference_batch_size = 32 * n_gpus

    print('-'*50)
    print(f"train data = {len(train_data)} samples")
    print(f"valid data = {len(valid_data)} samples")
    print(f"test data = {len(test_data)} samples") 
    print('-'*50)
    print(f"training on {get_sequence_len_stats(train_data, tokenizer, max_length, input_type)}% samples with whole sequence")
    print(f"validating on {get_sequence_len_stats(valid_data, tokenizer, max_length, input_type)}% samples with whole sequence")
    print(f"testing on {get_sequence_len_stats(test_data, tokenizer, max_length, input_type)}% samples with whole sequence")
    print('-'*50)

    print("labels statistics on training set:")
    print("Mean:", train_labels_mean)
    print("Standard deviation:", train_labels_std)
    print("Max:", train_labels_max)
    print("Min:", train_labels_min)
    print("-"*50)

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

    # instantiate the model
    model = Predictor(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling, model_name=model_name)

    device_ids = [d for d in range(torch.cuda.device_count())]

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model.to(device)
        
    if model_name in ["llmprop_finetune", "matbert_finetune"]:
        pretrained_model_path = f"checkpoints/se_atom/best_llmprop-mofseq_checkpoint_for_MOF_SE_atom_prediction.pt"
   
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device(device)), strict=False)
        else:
            model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device(device)), strict=False) 
            model.to(device)

    # print the model parameters
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters = {model_trainable_params}")

    # create dataloaders
    print("Start creating train dataloader...")
    print(train_batch_size, inference_batch_size)
    start = time.time()
    train_dataloader = create_dataloaders(
        tokenizer, 
        train_data, 
        max_length, 
        train_batch_size, 
        property_value=property_name, 
        pooling=pooling, 
        normalize=True, 
        normalizer=normalizer_type,
        input_type=input_type,
        preprocessing_strategy=preprocessing_strategy
    )
    print('Train dataloader complete! Used', time_format(time.time()-start))
    
    print("Start creating valid dataloader...")
    start = time.time()
    valid_dataloader = create_dataloaders(
        tokenizer, 
        valid_data, 
        max_length, 
        inference_batch_size, 
        property_value=property_name, 
        pooling=pooling,
        normalize=False,
        input_type=input_type,
        preprocessing_strategy=preprocessing_strategy
    )
    print('Valid dataloader complete! Used', time_format(time.time()-start))

    print("Start creating test dataloader...")
    start = time.time()
    test_dataloader = create_dataloaders(
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
    print('Test dataloader complete! Used', time_format(time.time()-start))

    # define the optimizer
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr = learning_rate
        )
   
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=learn_rate
        )

    # set up the scheduler
    total_training_steps = len(train_dataloader) * epochs 
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup( #get_linear_schedule_with_warmup
            optimizer,
            num_warmup_steps= warmup_steps, #steps_ratio*total_training_steps,
            num_training_steps=total_training_steps 
        )
    
    # from <https://github.com/usnistgov/alignn/blob/main/alignn/train.py>
    elif scheduler_type == 'onecycle': 
        steps_per_epoch = len(train_dataloader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    
    elif scheduler_type == 'step':
         # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=warmup_steps
        )
    
    elif scheduler_type == 'lambda':
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    
    print("======= Training ... ========")
    torch.cuda.empty_cache()
    training_stats, validation_predictions = train(model, optimizer, scheduler, bce_loss_function, mae_loss_function, 
        epochs, train_dataloader, valid_dataloader, device, normalizer=normalizer_type)
    
    print("======= Evaluating on test set ========")
    best_model_path = f"{checkpoints_directory}/best_{model_name}-{input_type}_checkpoint_for_MOF_{property_name}_prediction_{task_name}_{preprocessing_strategy}_{max_length}_tokens_{epochs}_epochs_{learning_rate}_{drop_rate}_training_size_{int(training_size*100)}%.pt"
    best_model = Predictor(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling, model_name=model_name)

    if torch.cuda.is_available():
        best_model = nn.DataParallel(best_model, device_ids=device_ids).cuda()
    else:
        best_model.to(device)

    if isinstance(best_model, nn.DataParallel):
        best_model.module.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False)
    else:
        best_model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False) 
        best_model.to(device)

    # averaging the results over 5 runs
    predictions = []
    test_results = []

    for i in range(5):
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)  
        predictions_list, test_performance = evaluate(best_model, mae_loss_function, test_dataloader, train_labels_mean, train_labels_std, property_name, device, task_name, normalizer=normalizer_type)
        predictions.append(predictions_list)
        test_results.append(test_performance)
    
    averaged_predictions = np.mean(np.array(predictions), axis=0)
    averaged_loss = np.mean(np.array(test_results))
    confidence_score = stdev(np.array(test_results))

    print('#'*50)
    print('The averaged test performance over 5 runs:', averaged_loss)
    print('The standard deviation:', confidence_score)
    print('#'*50)

    # save the averaged predictions
    test_predictions = {f"mof_name":list(test_data['mof_name']), f"actual_{property_name}":list(test_data[property_name]), f"predicted_{property_name}":averaged_predictions}
    saveCSV(pd.DataFrame(test_predictions), f"{statistics_directory}/{model_name}-{input_type}_test_stats_for_{property_name}_{task_name}_{preprocessing_strategy}_{max_length}_tokens_{epochs}_epochs_{learning_rate}_{drop_rate}_training_size_{int(training_size*100)}%.csv")