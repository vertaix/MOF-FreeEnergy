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
from llmprop_multimodal_model import Predictor
from llmprop_utils import *
from llmprop_multimodal_dataset_unsupervised import *
from llmprop_multimodal_args_parser import *

# for metrics
from torchmetrics.classification import BinaryAUROC
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr, kendalltau
from statistics import stdev

# for Weight&Biases
import wandb
from wandb import AlertLevel
from datetime import timedelta

import subprocess

def predict(
    model, 
    mae_loss_function, 
    test_dataloader, 
    train_labels_mean, 
    train_labels_std, 
    property_name,
    device,
    task_name,
    normalizer="z_norm",
    preprocessing_strategy=None,
    train_labels_min=None,
    train_labels_max=None
):
    """
    Optimized prediction function with better memory management and faster processing.
    """
    predict_start_time = time.time()
    model.eval()
    
    # Pre-calculate expected total number of samples to avoid list appends
    total_samples = len(test_dataloader.dataset)
    
    # Preallocate numpy arrays for results
    # Get the output shape by running a single batch through
    with torch.inference_mode():  # More efficient than torch.no_grad()
        sample_batch = next(iter(test_dataloader))
        if preprocessing_strategy == 'xVal':
            batch_inputs, batch_masks, batch_x_num = tuple(b.to(device) for b in sample_batch)
            sample_embeddings, sample_predictions = model(batch_inputs, batch_masks, x_num=batch_x_num)
        else:
            batch_inputs, batch_masks = tuple(b.to(device) for b in sample_batch)
            sample_embeddings, sample_predictions = model(batch_inputs, batch_masks)
    
    # Get embedding dimension
    embedding_dim = sample_embeddings.shape[1]
    
    # Preallocate arrays
    all_predictions = np.zeros(total_samples)
    all_embeddings = np.zeros((total_samples, embedding_dim))
    
    # Process batches
    idx = 0
    with torch.inference_mode():  # More efficient than torch.no_grad()
        for batch in tqdm(test_dataloader, desc="Prediction progress"):
            if preprocessing_strategy == 'xVal':
                batch_inputs, batch_masks, batch_x_num = tuple(b.to(device) for b in batch)
                embeddings, predictions = model(batch_inputs, batch_masks, x_num=batch_x_num)
            else:
                batch_inputs, batch_masks = tuple(b.to(device) for b in batch)
                embeddings, predictions = model(batch_inputs, batch_masks)

            # Apply denormalization based on task type
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
            
            # Process batch at once instead of looping through samples
            batch_size = predictions.shape[0]
            
            # Handle predictions (taking first element of each prediction)
            pred_np = predictions_denorm.detach().cpu().numpy()[:, 0]
            all_predictions[idx:idx+batch_size] = pred_np
            
            # Handle embeddings
            emb_np = embeddings.detach().cpu().numpy()
            all_embeddings[idx:idx+batch_size] = emb_np
            
            idx += batch_size
    
    predict_ending_time = time.time()
    predicting_time = time_format(predict_ending_time-predict_start_time)
    print(f"Testing took {predicting_time} \n")
    
    return all_embeddings.tolist(), all_predictions.tolist()

if __name__ == "__main__":
    # Start overall timing
    start_time = time.time()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        print(f'Number of available devices: {n_gpus}')
        print(f'Current device is: {torch.cuda.current_device()}')
        print(f"Evaluating on {n_gpus} GPUs!")
        print('-'*50)
        
        # Enable cudnn benchmarking for speed (if not deterministic)
        torch.backends.cudnn.benchmark = True
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")
        n_gpus = 0
    
    # Parse Arguments
    args = args_parser()
    config = vars(args)
    
    # Extract configuration parameters
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
    iteration_n = 0
    additional_samples_type = "top_10k"
    
    if model_name == "matbert":
        pooling = None
    
    # Print configuration for logging
    print(f"Configuration: {config}")
    print(f"Input type: {input_type}")
    print(f"Model: {model_name}")
    print(f"Task: {task_name}")
    print('-'*50)
    
    # Load data only once
    print("Loading data...")
    data_loading_start = time.time()
    
    # Load data using more efficient methods
    def concatenate_and_shuffle(df_1, df_2):
        concatenated_df = pd.concat([df_1, df_2], ignore_index=True)
        shuffled_df = concatenated_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return shuffled_df

    train_data = pd.read_csv(f"/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/data/properties/{property_name.lower()}/train.csv")
    candidates_data_path ="/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/data/properties/fe_atom/iterative_training/candidates_mofs_iteration_0.csv"
    candidates_data = pd.read_csv(candidates_data_path)
    
    # Drop duplicates and NaN values
    if input_type in ["mof_name", "mofkey", "mofid_v1"]:
        train_data = train_data.dropna(subset=[input_type]).reset_index(drop=True)
        candidates_data = candidates_data.dropna(subset=[input_type]).reset_index(drop=True)
    
    data_loading_end = time.time()
    print(f"Data loading took {time_format(data_loading_end - data_loading_start)}")
    print(f"Train data size: {len(train_data)}")
    print(f"Candidates data size: {len(candidates_data)}")
    print('-'*50)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_start = time.time()
    
    if tokenizer_name == 't5_tokenizer':
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
    elif tokenizer_name == 'modified':
        tokenizer = AutoTokenizer.from_pretrained("/n/fs/rnspace/projects/vertaix/LLM-Prop/tokenizers/new_pretrained_t5_tokenizer_on_modified_oneC4files_and_mp22_web_descriptions_32k_vocab")
    elif tokenizer_name == 'matbert_tokenizer':
        tokenizer = BertTokenizerFast.from_pretrained("/n/fs/rnspace/projects/vertaix/MatBERT/matbert-base-uncased", do_lower_case=True)
    
    # Add special tokens - all at once for efficiency
    special_tokens = []
    if pooling == 'cls':
        special_tokens.append("[CLS]")
    if preprocessing_strategy == "xVal":
        special_tokens.append("[NUM]")
    if input_type == "mof_name_and_cif_string":
        special_tokens.append("[SEP]")
    elif input_type == "combined_mof_str":
        special_tokens.extend(["<mofname>", "</mofname>", "<mofid>", "</mofid>", "<mofkey>", "</mofkey>"])
    elif input_type == "mofname_and_mofid":
        special_tokens.extend(["<mofname>", "</mofname>", "<mofid>", "</mofid>"])
    
    if special_tokens:
        tokenizer.add_tokens(special_tokens)
    
    tokenizer_end = time.time()
    print(f"Tokenizer loading took {time_format(tokenizer_end - tokenizer_start)}")
    print('-'*50)
    
    # Process data
    print("Processing data...")
    data_processing_start = time.time()
    
    # Process input data based on type
    if input_type == "mofkey":
        # Use vectorized operations instead of apply where possible
        train_data[input_type] = train_data[input_type].apply(clean_mofkey)
        candidates_data[input_type] = candidates_data[input_type].apply(clean_mofkey)
    elif input_type == "mofid_v1":
        train_data[input_type] = train_data[input_type].apply(clean_mofid)
        candidates_data[input_type] = candidates_data[input_type].apply(clean_mofid)
    elif input_type == "combined_mof_str":
        train_data = combine_mof_string_representations(train_data, tokenizer, input_type, max_length=max_length)
        candidates_data = combine_mof_string_representations(candidates_data, tokenizer, input_type, max_length=max_length)
    elif input_type == "mofname_and_mofid":
        candidates_data = combined_mofname_and_mofid(candidates_data, tokenizer, input_type, max_length=max_length)
    
    # Remove duplicates
    candidates_data = candidates_data.drop_duplicates(subset=[input_type]).reset_index(drop=True)
    
    # Process labels
    train_labels_array = np.array(train_data[property_name])
    train_labels_mean = torch.tensor(np.mean(train_labels_array), dtype=torch.float32)
    train_labels_std = torch.tensor(np.std(train_labels_array), dtype=torch.float32)
    train_labels_min = torch.tensor(np.min(train_labels_array), dtype=torch.float32)
    train_labels_max = torch.tensor(np.max(train_labels_array), dtype=torch.float32)
    
    # Apply preprocessing strategy
    if preprocessing_strategy == "xVal":
        # Use tqdm for progress tracking
        tqdm.pandas(desc="Processing train data")
        train_data['list_of_numbers_in_input'] = train_data[input_type].progress_apply(get_numbers_in_a_sentence)
        train_data[input_type] = train_data[input_type].progress_apply(replace_numbers_with_num)
    
    data_processing_end = time.time()
    print(f"Data processing took {time_format(data_processing_end - data_processing_start)}")
    print('-'*50)
    
    # Initialize loss functions
    mae_loss_function = nn.L1Loss()
    
    # Set freeze flag for model parameters
    freeze = False
    
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Fix for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Process candidate data
    for data in [candidates_data]:
        # Determine sequence length and batch size
        if input_type in ["mof_name", "mofkey"]:
            max_length = get_max_len(data, tokenizer, input_type)
            print(f'The longest composition has {max_length} tokens')
        
        print(f'Max length: {max_length}')
        
        # Set batch size based on sequence length
        if max_length <= 888:
            inference_batch_size = 1024 * max(1, n_gpus)
        elif max_length == 1024:
            inference_batch_size = 512 * max(1, n_gpus)
        elif max_length == 2000:
            inference_batch_size = 128 * max(1, n_gpus)
        else:
            # For very long sequences, use smaller batch size
            inference_batch_size = 128 * max(1, n_gpus)
        
        print(f"Using inference batch size: {inference_batch_size}")
        
        # Display label statistics
        print("Labels statistics on training set:")
        print(f"Mean: {train_labels_mean.item()}")
        print(f"Standard deviation: {train_labels_std.item()}")
        print(f"Max: {train_labels_max.item()}")
        print(f"Min: {train_labels_min.item()}")
        print("-"*50)
        
        print("======= Creating model and performing inference ========")
        model_start_time = time.time()
        
        # Load pretrained model
        if model_name in ["llmprop", "llmprop_finetune"]:
            base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
            base_model_output_size = 512
        elif model_name == "matbert":
            base_model = BertModel.from_pretrained("/n/fs/rnspace/projects/vertaix/MatBERT/matbert-base-uncased")
            base_model_output_size = 768
        
        # Freeze model parameters if specified
        if freeze:
            for param in base_model.parameters():
                param.requires_grad = False
        
        # Resize token embeddings
        base_model.resize_token_embeddings(len(tokenizer))
        
        # Load checkpoint
        best_model_path = f'/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/checkpoints/1m_mof/mofbench_llmprop_finetune_iteration_0_best_checkpoint_for_FE_atom_regression_mofname_and_mofid_none_2000_tokens_200_epochs_0.001_0.2_100.0%_no_outliers.pt'
         
        # Initialize model
        best_model = Predictor(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling, model_name=model_name)
        
        # Move model to device
        device_ids = list(range(n_gpus)) if n_gpus > 0 else None
        
        if torch.cuda.is_available() and n_gpus > 0:
            best_model = nn.DataParallel(best_model, device_ids=device_ids).cuda()
        else:
            best_model.to(device)
        
        # Load state dict
        print(f"Loading model from {best_model_path}")
        if isinstance(best_model, nn.DataParallel):
            best_model.module.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
        else:
            best_model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
            best_model.to(device)
        
        model_end_time = time.time()
        print(f"Model creation took {time_format(model_end_time - model_start_time)}")
        
        # Create dataloaders
        dataloader_start_time = time.time()
        
        # Create optimized dataloader
        dataloader = create_dataloaders(
            tokenizer, 
            data, 
            max_length, 
            inference_batch_size, 
            property_value=property_name, 
            pooling=pooling,
            normalize=False,
            input_type=input_type,
            preprocessing_strategy=preprocessing_strategy,
            num_workers=4,  # Use multiple workers for faster data loading
            pin_memory=True  # Pin memory for faster GPU transfer
        )
        
        dataloader_end_time = time.time()
        print(f"Dataloader creation took {time_format(dataloader_end_time - dataloader_start_time)}")
        
        # Run prediction
        print("Running predictions...")
        embeddings_list, predictions_list = predict(
            best_model, 
            mae_loss_function, 
            dataloader, 
            train_labels_mean, 
            train_labels_std, 
            property_name,
            device,
            task_name, 
            normalizer=normalizer_type,
            preprocessing_strategy=preprocessing_strategy,
            train_labels_min=train_labels_min,
            train_labels_max=train_labels_max
        )
        
        # Save predictions
        save_start_time = time.time()
        
        # Create output DataFrame
        predictions_data = pd.DataFrame()
        predictions_data['mof_name'] = data['mof_name','mofid_v1', 'SE_atom']
        predictions_data[f'predicted_FE_atom'] = predictions_list
        
        # Save to CSV
        output_csv_path = f"/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/data/properties/fe_atom/iterative_training/candidates_mofs_iteration_0_2000.csv"
        
        predictions_data.to_csv(output_csv_path, index=False)
        
        # Save embeddings
        output_npy_path = f"/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/data/properties/fe_atom/iterative_training/candidates_embeddings_iteration_{iteration_n}_{additional_samples_type}_{max_length}.npy"
        
        # Convert to numpy array and save
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        np.save(output_npy_path, embeddings_array, allow_pickle=False)
        
        save_end_time = time.time()
        print(f"Saving results took {time_format(save_end_time - save_start_time)}")
        
        print(f"Results saved to:")
        print(f"  - {output_csv_path}")
        print(f"  - {output_npy_path}")
    
    # End overall timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal script execution time: {time_format(total_time)}")