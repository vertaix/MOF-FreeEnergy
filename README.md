# [Highly Accurate and Fast Prediction of MOF Free Energy Via Machine Learning](https://chemrxiv.org/engage/chemrxiv/article-details/686fedbc43bc52e4ec4edfc6)
This repository contains the implementation of our proposed machine learning approach for MOF Free Energy Prediction.

<p align="center" width="100%">
    <img src="figures/MOFseq_full.png" alt="image" width="100%" height="auto" style="display: inline-block;">
    <!-- <img src="figures/mofseq_embeddings.png" alt="image" width="40%" height="auto" style="display: inline-block;"> -->
    <br>
    <em> Schematic representation of MOFseq </em>
</p>

For more details, please read our paper, [Highly Accurate and Fast Prediction of MOF Free Energy Via Machine Learning](https://chemrxiv.org/engage/chemrxiv/article-details/686fedbc43bc52e4ec4edfc6)

## Installation
You can install MOF-FreeEnergy by following these steps:
```
git clone https://github.com/vertaix/MOF-FreeEnergy.git
cd MOF-FreeEnergy
conda create -n <environment_name> requirement.txt
conda activate <environment_name>
```
## Usage
### Preparing the data
First download the preprocessed data from this [link](https://drive.google.com/drive/folders/18joRpZCNW8guhHTtjZJYA0IAsE3-Wm-7) and save the `data` folder in the root directory. The `mofseq` part in the data files is the MOF representation with `2000 tokens` sequence length that is ready to use. To adjust the sequence length, please refer to this [notebook](data_preparation.ipynb) and change the `paths` and the `max_len` value accordingly. 

### Pretraining
We first pretrain on strain energy by running:
```python
python src/llmprop_train.py \
    --model_name llmprop \
    --property_name SE_atom \
    --dr 0.2 \
    --lr 1e-3 \
    --max_len 2000 \
    --epochs 100 \
    --train_bs 64 \
    --inference_bs 512
```
### Finetuning
Then we finetune on free energy by running:
```python
python src/llmprop_train.py \
    --model_name llmprop_finetune \
    --property_name FE_atom \
    --dr 0.2 \
    --lr 1e-3 \
    --max_len 2000 \
    --epochs 200 \
    --train_bs 64 \
    --inference_bs 512
```
### Evaluating
For evaluation run:
```python
python src/llmprop_evaluate.py --inference_bs 512
```

## Citation
```bibtex
@article{rubungo2025highly,
  title={Highly Accurate and Fast Prediction of MOF Free Energy Via Machine Learning},
  author={Rubungo, Andre Niyongabo and Fajardo-Rojas, Fernando and G{\'o}mez-Gualdr{\'o}n, Diego and Dieng, Adji Bousso},
  year={2025}
}
```