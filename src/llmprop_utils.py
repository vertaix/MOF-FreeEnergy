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


def generate_mofseq(
    tokenizer,
    df=None,
    mofname=None,
    mofid=None,
    space_group=None,
    input_type="mofseq-1",
    max_length=2000,
):
    def _to_list(x):
        if x is None:
            return None
        return [x] if isinstance(x, str) else list(x)

    def _build_sequence(m_name, m_id, sg=None):
        mof_name = f"<mofname>{m_name}</mofname>"
        mofid_clean = clean_mofid(m_id)
        mof_id = f"<mofid>{mofid_clean}</mofid>"

        use_space_group = input_type == "mofseq-2"
        space_group_str = (
            f"<spacegroup>{sg}</spacegroup>"
            if use_space_group and sg is not None
            else ""
        )

        combined = f"{mof_name}{space_group_str}{mof_id}"

        if len(tokenizer.tokenize(combined)) < max_length:
            return combined

        prefix = f"{mof_name}{space_group_str}"
        prefix_len = len(tokenizer.tokenize(prefix))

        reserved_tokens = 7 if use_space_group else 5
        mofid_len = max_length - prefix_len - reserved_tokens

        truncated_mofid = truncate_sentence(tokenizer, mofid_len, mofid_clean)
        return f"{prefix}<mofid>{truncated_mofid}</mofid>"

    # DataFrame mode
    if df is not None:
        combined_mof_strs = []

        for _, row in df.iterrows():
            sg = row["space_group"] if input_type == "mofseq-2" else None
            combined_mof_strs.append(
                _build_sequence(
                    row["mof_name"],
                    row["mofid_v1"],
                    sg,
                )
            )

        df[input_type] = combined_mof_strs
        return df

    # String/list mode
    mofname = _to_list(mofname)
    mofid = _to_list(mofid)
    space_group = _to_list(space_group)

    if mofname is None or mofid is None:
        raise ValueError("Provide either df or both mofname and mofid.")

    if len(mofname) != len(mofid):
        raise ValueError("mofname and mofid must have the same length.")

    if input_type == "mofseq-2":
        if space_group is None:
            raise ValueError("space_group must be provided when input_type='mofseq-2'.")
        if len(space_group) != len(mofname):
            raise ValueError("space_group must have the same length as mofname and mofid.")
    else:
        space_group = [None] * len(mofname)

    return [
        _build_sequence(m_name, m_id, sg)
        for m_name, m_id, sg in zip(mofname, mofid, space_group)
    ]