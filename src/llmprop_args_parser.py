import argparse

def args_parser():

    parser = argparse.ArgumentParser(description='LLM-Prop')
    
    parser.add_argument('--epochs',
                        help='Number of epochs',
                        type=int,
                        default=200)
    parser.add_argument('--train_bs',
                        help='Batch size',
                        type=int,
                        default=64)
    parser.add_argument('--inference_bs',
                        help='Batch size',
                        type=int,
                        default=1024)
    parser.add_argument('--lr',
                        help='Learning rate',
                        type=float,
                        default=0.001)
    parser.add_argument('--max_len',
                        help='Max input sequence length',
                        type=int,
                        default=888)
    parser.add_argument('--dr',
                        help='Drop rate',
                        type=float,
                        default=0.5)
    parser.add_argument('--warmup_steps',
                        help='Warmpup steps',
                        type=int,
                        default=30000)
    parser.add_argument('--preprocessing_strategy',
                        help='Data preprocessing technique: "none", "bond_lengths_replaced_with_num", "bond_angles_replaced_with_ang", "no_stopwords", or "no_stopwords_and_lengths_and_angles_replaced"',
                        type=str,
                        default="no_stopwords_and_lengths_and_angles_replaced")
    parser.add_argument('--tokenizer',
                        help='Tokenizer name: "t5_tokenizer" or "modified"',
                        type=str,
                        default="modified")
    parser.add_argument('--pooling', 
                        help='Pooling method. "cls" or "mean"',
                        type=str,
                        default="cls")
    parser.add_argument('--normalizer', 
                        help='Labels scaling technique. "z_norm", "mm_norm", or "ls_norm"',
                        type=str,
                        default="z_norm") 
    parser.add_argument('--scheduler', 
                        help='Learning rate scheduling technique. "linear", "onecycle", "step", or "lambda" (no scheduling))',
                        type=str,
                        default="onecycle")
    parser.add_argument('--property_name', 
                        help='The name of the property to predict. "band_gap", "volume", or "is_gap_direct"',
                        type=str,
                        default="band_gap")
    parser.add_argument('--optimizer', 
                        help='Optimizer type. "adamw" or "sgd"',
                        type=str,
                        default="adamw")
    parser.add_argument('--task_name', 
                        help='the name of the task: "regression" if propert_name is band_gap or volume or "classification" if property_name is is_gap_direct',
                        type=str,
                        default="regression")
    parser.add_argument('--data_path',
                        help="the path to the data",
                        type=str,
                        default="/n/fs/rnspace/projects/vertaix/nlp4matbench/data")                    
    parser.add_argument('--train_data_path',
                        help="the path to the training data",
                        type=str,
                        default="data/samples/textedge_prop_mp22_train.csv")
    parser.add_argument('--valid_data_path',
                        help="the path to the valid data",
                        type=str,
                        default="data/samples/textedge_prop_mp22_valid.csv")
    parser.add_argument('--test_data_path',
                        help="the path to the test data",
                        type=str,
                        default="data/samples/textedge_prop_mp22_test.csv")
    parser.add_argument('--checkpoint',
                        help="the path to the the best checkpoint for evaluation",
                        type=str,
                        default="")
    parser.add_argument('--input_type',
                        help="description, structure, or composition",
                        type=str,
                        default="description")
    parser.add_argument('--dataset_name',
                        help="mp, ...",
                        type=str,
                        default="mp")
    parser.add_argument('--model_name',
                        help="llmprop, ...",
                        type=str,
                        default="llmprop")
    parser.add_argument('--regressor',
                        help="linear, ...",
                        type=str,
                        default="linear")
    parser.add_argument('--loss',
                        help="mae, ...",
                        type=str,
                        default="mae")
    parser.add_argument('--training_size',
                        help="0.2, 0.4, ..., 1.0",
                        type=float,
                        default=1.0)
    parser.add_argument('--iteration_no',
                        help="the n_th self_training iteration",
                        type=int,
                        default=0)
    parser.add_argument('--additional_samples_type',
                        help="the type of additional samples to be used for self-training. 'bot_10k' or 'top_10k'",
                        type=str,
                        default="bot_10k")
    args = parser.parse_args()
    
    return args
