# [Highly Accurate and Fast Prediction of MOF Free Energy Via Machine Learning](https://pubs.acs.org/doi/full/10.1021/jacs.5c13960)
This repository contains the implementation of our proposed machine learning approach for MOF Free Energy Prediction.

<p align="center" width="100%">
    <img src="figures/MOFseq_full.png" alt="image" width="100%" height="auto" style="display: inline-block;">
    <!-- <img src="figures/mofseq_embeddings.png" alt="image" width="40%" height="auto" style="display: inline-block;"> -->
    <br>
    <em> Schematic representation of MOFseq </em>
</p>

For more details, please read our paper, [Highly Accurate and Fast Prediction of MOF Free Energy Via Machine Learning](https://pubs.acs.org/doi/full/10.1021/jacs.5c13960)

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

### Inference
Follow the below example to predict the free energy of one or few samples. For more details, please check this [inference notebook](/n/fs/rnspace/projects/vertaix/MOF-FreeEnergy/inference.ipynb):

```python
from src.llmprop_inference import predict

mofname = "SR_nkc_v1-4c_Cu_1_Ch_v2-4c_1anC_Ch_v3-3c_B_Ch_2B_fused_Ch"
mofid = "[Cu][Cu].[O-]C(=O)c1ccc2c(c1)ccc(c2)c1cc2-c3ccc4c(c3)ccc(c4)C3(c4ccc5c(c4)ccc(-c4cc(-c6cc7ccc(-c(c1)c2)cc7cc6)cc(c4)c1ccc2c(c1)ccc(c2)C(=O)[O-])c5)c1ccc2c(c1)ccc(-c1cc(-c4ccc5cc(-c6cc(-c7cc8ccc3cc8cc7)cc(c6)c3ccc6c(c3)ccc(c6)C(=O)[O-])ccc5c4)cc(c1)c1ccc3c(c1)ccc(c3)C(=O)[O-])c2 MOFid-v1.TIMEOUT.cat0.NO_REF;SR_nkc_v1-4c_Cu_1_Ch_v2-4c_1anC_Ch_v3-3c_B_Ch_2B_fused_Ch"
prop_name = "FE_atom" # options: "FE_atom" for free energy, "SE_atom" for strain energy

embeddings, predictions, predicting_time = predict(
    input_mofname=mofname,
    input_mofid=mofid,
    property_name=prop_name
    )

print("-"*50)
print(f'predicted {prop_name}:', predictions)
print('inference time (s): ', predicting_time)
```

## Citation
```bibtex
@article{NiyongaboRubungo2025,
author={Niyongabo Rubungo, Andre and Fajardo-Rojas, Fernando and G{\'o}mez-Gualdr{\'o}n, Diego A. and Dieng, Adji Bousso},
title={Highly Accurate and Fast Prediction of MOF Free Energy via Machine Learning},
journal={Journal of the American Chemical Society},
year={2025},
month={Dec},
day={16},
publisher={American Chemical Society},
issn={0002-7863},
doi={10.1021/jacs.5c13960},
url={https://doi.org/10.1021/jacs.5c13960}
}
```