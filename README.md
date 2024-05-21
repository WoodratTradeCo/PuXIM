# Purified-ZS-SBIR
## Image Examples
<div align=center>
<img width="800" alt="1696749034041" src="https://github.com/WoodratTradeCo/Purified-ZS-SBIR/blob/main/figures/0.png">
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
