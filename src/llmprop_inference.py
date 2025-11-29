#import necessary libraries
import time
import re
import tarfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5EncoderModel, AutoTokenizer
from src.llmprop_model import Predictor
from src.llmprop_utils import *

#disable warnings
import warnings
warnings.filterwarnings("ignore")


def predict(
    trained_model_ckpt: str | None = None, # path to your trained model checkpoint (default:None, can be retrieved from the property_name)
    # train_data_path: str | None = None, # path to your training data csv file
    input_mofseq: str | list[str] | None = None, # one mofseq string or a list of mofseq strings 
    input_mofname: str | list[str] | None = None, # one mofname string or a list of mofname strings 
    input_mofid: str | list[str] | None = None, # one mofid string or a list of mofid strings
    batch_size: int = 64, # batch size for prediction
    property_name: str = "FE_atom", # property name to predict: FE_atom or SE_atom for now
    normalizer: str = "z_norm", # normalization method used in training: z_norm, mm_norm, ls_norm, no_norm
    preprocessing_strategy: str | None = None,
    max_length: int = 2000, # max length for mofseq generation and tokenization
    freeze: bool = False, # whether to freeze the base model parameters
    task_name: str = "regression" # task name: regression or classification
):
    predict_start_time = time.time()
    
    # load tokenizer
    # print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    tokenizer.add_tokens(["[CLS]","<mofname>", "</mofname>", "<mofid>", "</mofid>"]) # special tokens for generating mofseq from mofname and mofid
    
    # handling input types
    if isinstance(input_mofseq, str):
        input_mofseq = [input_mofseq]
    if isinstance(input_mofname, str):
        input_mofname = [input_mofname]
    if isinstance(input_mofid, str):
        input_mofid = [input_mofid]
        
    if input_mofseq is None and input_mofname is None and input_mofid is None:
        raise ValueError("Please provide at least one type of input: mofseq, mofname, or mofid.")
    if input_mofseq is not None:
        if not isinstance(input_mofseq, list):
            raise ValueError("input_mofseq should be a string or a list of strings.")
    if input_mofname is not None:
        if not isinstance(input_mofname, list):
            raise ValueError("input_mofname should be a string or a list of strings.")
    if input_mofid is not None:
        if not isinstance(input_mofid, list):
            raise ValueError("input_mofid should be a string or a list of strings.")
    
    if input_mofseq is None and input_mofname is not None and input_mofid is not None:
        input_mofseq = generate_mofseq_str(input_mofname, input_mofid, tokenizer, max_length=max_length)
    
    if trained_model_ckpt is None:
        if property_name is not None:
            trained_model_ckpt = f'checkpoints/{property_name.lower()}/best_llmprop-mofseq_checkpoint_for_MOF_{property_name}_prediction.pt'
        else:
            raise ValueError(f"Property name {property_name} not recognized. Please provide a valid property name or a path to your trained model checkpoint.")
    
    # process train data labels for denormalization
    train_data_config = readJSON(f"checkpoints/{property_name.lower()}/mofseq_config.json")
    train_labels_mean = torch.tensor(train_data_config['train_data_info'][f'mean_{property_name}'], dtype=torch.float32)
    train_labels_std = torch.tensor(train_data_config['train_data_info'][f'std_{property_name}'], dtype=torch.float32)
    train_labels_min = torch.tensor(train_data_config['train_data_info'][f'min_{property_name}'], dtype=torch.float32)
    train_labels_max = torch.tensor(train_data_config['train_data_info'][f'max_{property_name}'], dtype=torch.float32) 
    
    # create a dataloader
    encoded_input = tokenizer(text=["[CLS] " + str(mof_str) for mof_str in input_mofseq],
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length,
                                    return_attention_mask=True)
    input_ids, attention_masks = encoded_input['input_ids'], encoded_input['attention_mask']
    input_tensor = torch.tensor(input_ids)
    mask_tensor = torch.tensor(attention_masks)
    
    dataloader = DataLoader(TensorDataset(input_tensor, mask_tensor), batch_size=batch_size, shuffle=False)
    
    #======= Creating model and performing inference ========"
    model_start_time = time.time()
    
    base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
    base_model_output_size = 512

    if freeze:
        for param in base_model.parameters():
            param.requires_grad = False

    base_model.resize_token_embeddings(len(tokenizer))

    # Initialize model
    model = Predictor(base_model, base_model_output_size, drop_rate=0.2, pooling='cls', model_name='llmprop')
    
    # Move model to device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        print(f'Number of available devices: {n_gpus}', f'Current device is: {torch.cuda.current_device()}', f"predicting with {n_gpus} GPUs!")
        
        # Enable cudnn benchmarking for speed (if not deterministic)
        torch.backends.cudnn.benchmark = True
    else:
        # print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        device = torch.device("cpu")
        n_gpus = 0
    
    device_ids = list(range(n_gpus)) if n_gpus > 0 else None
    
    if torch.cuda.is_available() and n_gpus > 0:
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model.to(device)
    
    # Load state dict
    # print(f"Loading model from {trained_model_ckpt}")
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(trained_model_ckpt, map_location=device), strict=False)
    else:
        model.load_state_dict(torch.load(trained_model_ckpt, map_location=device), strict=False)
        model.to(device)
    
    model_end_time = time.time()
    print(f"Model creation took {time_format(model_end_time - model_start_time)}")
    
    # Pre-calculate expected total number of samples to avoid list appends
    total_samples = len(dataloader.dataset)
    
    # Preallocate numpy arrays for results
    # Get the output shape by running a single batch through
    model.eval()
    with torch.inference_mode():  # More efficient than torch.no_grad()
        sample_batch = next(iter(dataloader))
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
        for batch in tqdm(dataloader, desc="Prediction progress"):
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
    print(f"Predicting took {predicting_time} -> {predict_ending_time-predict_start_time} seconds")
    
    return all_embeddings.tolist(), all_predictions.tolist(), predict_ending_time-predict_start_time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict MOF properties using a trained LLM model.")
    parser.add_argument("--trained_model_ckpt", type=str, default=None, help="Path to the trained model checkpoint.")
    parser.add_argument("--input_mofseq", type=str, nargs='*', default=None, help="One or more MOF sequence strings.")
    parser.add_argument("--input_mofname", type=str, nargs='*', default=None, help="One or more MOF name strings.")
    parser.add_argument("--input_mofid", type=str, nargs='*', default=None, help="One or more MOF ID strings.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for prediction.")
    parser.add_argument("--property_name", type=str, default="FE_atom", help="Property name to predict (e.g., FE_atom).")
    parser.add_argument("--normalizer", type=str, default="z_norm", help="Normalization method used in training (e.g., z_norm).")
    parser.add_argument("--preprocessing_strategy", type=str, default=None, help="Preprocessing strategy if any.")
    parser.add_argument("--max_length", type=int, default=2000, help="Max length for MOF sequence generation and tokenization.")
    parser.add_argument("--freeze", type=bool, default=False, help="Whether to freeze the base model parameters.")
    parser.add_argument("--task_name", type=str, default="regression", help="Task name: regression or classification.")
    
    args = parser.parse_args()
    
    embeddings, predictions, used_time = predict(
        trained_model_ckpt=args.trained_model_ckpt,
        input_mofseq=args.input_mofseq,
        input_mofname=args.input_mofname,
        input_mofid=args.input_mofid,
        batch_size=args.batch_size,
        property_name=args.property_name,
        normalizer=args.normalizer,
        preprocessing_strategy=args.preprocessing_strategy,
        max_length=args.max_length
    )

    print("Predictions:", predictions, "\n")
    
    