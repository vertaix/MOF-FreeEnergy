import re
import json
import glob
import torch
import tarfile
import datetime
import numpy as np
import pandas as pd

def writeToJSON(data, where_to_save):
    """
    data: a dictionary that contains data to save
    where_to_save: the name of the file to write on
    """
    with open(where_to_save, "w", encoding="utf8") as outfile:
        json.dump(data, outfile)

def readJSON(input_file):
    """
    1. arguments
        input_file: a json file to read
    2. output
        a json objet in a form of a dictionary
    """
    with open(input_file, "r", encoding="utf-8", errors='ignore') as infile:
        json_object = json.load(infile, strict=False)
    return json_object

def writeTEXT(data, where_to_save):
    with open(where_to_save, "w", encoding="utf-8") as outfile:
        for d in data:
            outfile.write(str(d))
            outfile.write("\n")

def readTEXT_to_LIST(input_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        data = []
        for line in infile:
            data.append(line)
    return data

def saveCSV(df, where_to_save):
    df.to_csv(where_to_save, index=False)

def time_format(total_time):
    """
    Change the from seconds to hh:mm:ss
    """
    total_time_rounded = int(round((total_time)))
    total_time_final = str(datetime.timedelta(seconds=total_time_rounded))
    return total_time_final

def z_normalizer(labels):
    """ Implement a z-score normalization technique"""
    labels_mean = torch.mean(labels)
    labels_std = torch.std(labels)

    scaled_labels = (labels - labels_mean) / labels_std

    return scaled_labels

def z_denormalize(scaled_labels, labels_mean, labels_std):
    labels = (scaled_labels * labels_std) + labels_mean
    return labels

def min_max_scaling(labels):
    """ Implement a min-max normalization technique"""
    min_val = torch.min(labels)
    max_val = torch.max(labels)
    diff = max_val - min_val
    scaled_labels = (labels - min_val) / diff
    return scaled_labels

def mm_denormalize(scaled_labels, min_val, max_val):
    diff = max_val - min_val
    denorm_labels = (scaled_labels * diff) + min_val
    return denorm_labels

def log_scaling(labels):
    """ Implement log-scaling normalization technique"""
    scaled_labels = torch.log1p(labels)
    return scaled_labels

def ls_denormalize(scaled_labels):
    denorm_labels = torch.expm1(scaled_labels)
    return denorm_labels

def compressCheckpointsWithTar(filename):
    filename_for_tar = filename[0:-3]
    tar = tarfile.open(f"{filename_for_tar}.tar.gz", "w:gz")
    tar.add(filename)
    tar.close()

def decompressTarCheckpoints(tar_filename):
    tar = tarfile.open(tar_filename)
    tar.extractall()
    tar.close()

def get_sequence_len_stats(df, tokenizer, max_len, input_type):
    training_on = sum(1 for sent in df[input_type].apply(tokenizer.tokenize) if len(sent) <= max_len)
    return (training_on/len(df))*100

def get_max_len(df, tokenizer, input_type):
    max_len = max(len(sent) for sent in df[input_type].apply(tokenizer.tokenize))
    return max_len

def clean_mofkey(mofkey):
    mofkey = mofkey.replace('.MOFkey-v1','').replace('.TIMEOUT','')
    mofkey = mofkey.replace('.NO_REF','').replace('.ERROR','').replace('.UNKNOWN','')
    return mofkey

def clean_mofid(mofid):
    mofid = mofid.split(';')[0]
    mofid = mofid.split(' ')[1] + ' ' + mofid.split(' ')[0]
    mofid = mofid.replace('MOFid-v1.','').replace('.UNKNOWN','').replace('UNKNOWN.','')
    mofid = mofid.replace('.ERROR','').replace('.TIMEOUT','').replace('TIMEOUT.','')
    mofid = mofid.replace('.NO_REF','')
    return mofid

def truncate_sentence(tokenizer, max_tokens: int, sentence: str) -> str:
    tokens = tokenizer.tokenize(sentence)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
    return truncated_text

def generate_mofseq(df, tokenizer, mof_representation='mofseq', max_length=2000):
    combined_mof_strs = []
    
    for _, row in df.iterrows():
        mof_name = f"<mofname>{row['mof_name']}</mofname>"
        mofid = f"<mofid>{clean_mofid(row['mofid_v1'])}</mofid>"
        combined_mof_str = f"{mof_name}{mofid}"

        if len(tokenizer.tokenize(combined_mof_str)) < max_length:
            combined_mof_strs.append(combined_mof_str)
        else:
            part_1 = f"{mof_name}"
            part_1_len = len(tokenizer.tokenize(part_1))
            mofid_len = max_length - part_1_len - 5  # Reserve 5 tokens for special tokens
            truncated_mofid = truncate_sentence(tokenizer, mofid_len, clean_mofid(row['mofid_v1']))
            combined_mof_strs.append(f"{mof_name}<mofid>{truncated_mofid}</mofid>")

    df[mof_representation] = combined_mof_strs
    return df

def generate_mofseq_str(mofname: str, mofid: str, tokenizer, max_length: int = 2000):
    mof_seqs = []
    
    if isinstance(mofname, str):
        mofname = [mofname]
    if isinstance(mofid, str):
        mofid = [mofid]
    if len(mofname) != len(mofid):
        raise ValueError("mofname list and mofid list must have the same length.")
    
    for m_name, m_id in zip(mofname, mofid):
        mof_name = f"<mofname>{m_name}</mofname>"
        mof_id = f"<mofid>{clean_mofid(m_id)}</mofid>"

        combined_mof_str = f"{mof_name}{mof_id}"

        if len(tokenizer.tokenize(combined_mof_str)) < max_length:
            mof_seqs.append(combined_mof_str)
        else:
            part_1 = f"{mof_name}"
            part_1_len = len(tokenizer.tokenize(part_1))
            mofid_len = max_length - part_1_len - 7  # Reserve 7 tokens for special tokens
            
            truncated_mofid = truncate_sentence(tokenizer, mofid_len, clean_mofid(m_id))
            mof_seqs.append(f"{mof_name}<mofid>{truncated_mofid}</mofid>")
    return mof_seqs