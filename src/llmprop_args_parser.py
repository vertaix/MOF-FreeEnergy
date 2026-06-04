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
                        default=2000)
    parser.add_argument('--dr',
                        help='Drop rate',
                        type=float,
                        default=0.2)
    parser.add_argument('--warmup_steps',
                        help='Warmpup steps',
                        type=int,
                        default=30000)
    parser.add_argument('--preprocessing_strategy',
                        help='Data preprocessing technique: "none", "xVal"',
                        type=str,
                        default="none")
    parser.add_argument('--tokenizer',
                        help='Tokenizer name: "t5_tokenizer" ',
                        type=str,
                        default="t5_tokenizer")
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
                        help='The name of the property to predict.',
                        type=str,
                        default="FE_atom")
    parser.add_argument('--optimizer', 
                        help='Optimizer type. "adamw" or "sgd"',
                        type=str,
                        default="adamw")
    parser.add_argument('--task_name', 
                        help='"regression"',
                        type=str,
                        default="regression")
    parser.add_argument('--data_path',
                        help="the path to the data",
                        type=str,
                        default="data/")                    
    parser.add_argument('--checkpoint',
                        help="the path to the the best checkpoint for evaluation",
                        type=str,
                        default="")
    parser.add_argument('--input_type',
                        help="mof_name, mofid_v1, mofseq-1, or mofseq-2",
                        type=str,
                        default="mofseq-1")
    parser.add_argument('--model_name',
                        help="llmprop, llmprop_finetune",
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
    parser.add_argument('--training_ratio',
                        help="0.2, 0.4, ..., 1.0",
                        type=float,
                        default=1.0)
    parser.add_argument('pretraining_ckpt_path',
                        help="the path to the pretraining checkpoint for fine-tuning",
                        type=str,
                        default="")
    parser.add_argument('--checkpoint_path',
                        help="the path to the checkpoint for evaluation",
                        type=str,
                        default="")
    parser.add_argument('--config_path',
                        help="the path to the config file",
                        type=str,
                        default="")
    args = parser.parse_args()
    
    return args
