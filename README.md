# Purified Zero-Shot Sketch-Based Image Retrieval
![Python 3.6](https://img.shields.io/badge/python-3.8-green) ![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.10-green) ![MIT License](https://img.shields.io/badge/licence-MIT-green)

This repository is a purified zero-shot sketch-based image retrieval implementation for PuXIM (The paper is submitted to IEEE Transactions on Multimedia). 
PuXIM is a distraction-agnostic framework for precise semantic space training on a high semantic ambiguity or low-quality dataset. Our method outperforms SOTA methods across several datasets and achieves high mAP on the proposed high semantic ambiguity training dataset.

> Abstract: Sketches, characterized by sparse visual cues such as simple strokes, differ significantly from natural images, which contain complex elements like background, foreground, and texture. This misalignment poses substantial challenges for zero-shot sketch-based image retrieval (ZS-SBIR). Prior approaches match sketches to full images and tend to overlook redundant elements in natural images, leading to model distraction and semantic ambiguity. To tackle this issue, we introduce a distraction-agnostic framework, Purified Cross-Domain Matching (PuXIM), which operates on a straightforward principle: mask and match. We devise a Visual-cross-Linguistic (VxL) Sampler that generates linguistic masks based on semantic labels to obscure semantically irrelevant image features. Our novel contribution is the concept of Purified Masked Matching (PMM), comprising two processes: (1) reconstruction, compelling the image encoder to reconstruct the masked image feature, and (2) interaction, involving a transformer decoder that processes both sketch and masked image features to investigate cross-domain relationships for effective matching. Evaluated on the TU-Berlin, Sketchy, and QuickDraw datasets, PuXIM sets new benchmarks in performance. Importantly, the distraction-agnostic nature of the matching process renders PuXIM more conducive to training, enabling efficient adaptation to zero-shot scenarios with reduced data requirements and low data quality.

## Image Examples
<div align=center>
<img width="800" alt="1696749034041" src="https://github.com/WoodratTradeCo/Purified-ZS-SBIR/blob/main/figures/0.png">
</div>

<div align=center>
<img width="800" alt="1696749034043" src="https://github.com/WoodratTradeCo/Purified-ZS-SBIR/blob/main/figures/3.png">
</div>

## Model Architecture
<div align=center>
<img width="800" alt="1696749034040" src="https://github.com/WoodratTradeCo/Purified-ZS-SBIR/blob/main/figures/1.png">
</div>

## Dataset
The experiments are based on TU-Berlin, Sketchy, and QuickDraw datasets. We propose a high semantic ambiguity training set, the data can be achieved at https://pan.baidu.com/s/1wpqJ1Elu0Gi8iK3tIuH3Lg (sbir).
<div align=center>
<img width="400" alt="1696749034042" src="https://github.com/WoodratTradeCo/Purified-ZS-SBIR/blob/main/figures/2.png">
</div>

## Usage (How to Train Our PuXIM)

    # 1. Choose your workspace and download our repository.
    cd ${CUSTOMIZED_WORKSPACE}
    git clone https://github.com/WoodratTradeCo/Purified-ZS-SBIR
    # 2. Enter the directory.
    cd Purified-ZS-SBIR
    # 3. Clone our environment, and activate it.
    conda-env create --name ${CUSTOMIZED_ENVIRONMENT_NAME}
    conda activate ${CUSTOMIZED_ENVIRONMENT_NAME}
    # 4. Download dataset.
    # 5. Train our PuXIM. Please see details in our code annotations.
    # Please set the input arguments based on your case.
    # When the program starts running, a folder named 'results/${CUSTOMIZED_EXPERIMENT_NAME}' will be created automatically to save your log, checkpoint.
    python train.py
    --exp ${CUSTOMIZED_EXPERIMENT_NAME}
    --epoch ${CUSTOMIZED_EPOCH}
    --batch_size ${CUSTOMIZED_SIZE}   
    --num_workers ${CUSTOMIZED_NUMBER} 
    --gpu ${CUSTOMIZED_GPU_NUMBER}

## License
This project is released under the [MIT License](./LICENSE).
