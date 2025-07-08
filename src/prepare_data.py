import pandas as pd
from src.llmprop_utils import *

def prepare_data(input_file, output_file, tokenizer, mof_representation='mofseq', max_length=2000):
    """
    input_file: a csv file with mof_name, mofid_v1, and FE_atom (free energy) as columns
    output_file: a csv file with mof_name, mofid_v1, mofseq, and FE_atom (free energy) as columns
    tokenizer: any tokenizer object from the transformers library
    mof_representation: the representation of the MOF, default is 'mofseq'
    max_length: the maximum length of the input sequence, default is 2000
    """
    data_orig = pd.read_csv(input_file)
    data_new = generate_mofseq(data_orig, tokenizer, 
                                mof_representation=mof_representation, 
                                max_length=max_length)
    saveCSV(data_new, output_file)
    return data_new