# NeuralODEs for Brain Tumor Segmentation

Implementation of "A Neural Ordinary Differential Equation Model for Visualizing Deep Neural Network Behaviors in Multi-Parametric MRI based Glioma Segmentation"  
[Link to the paper](https://arxiv.org/abs/2203.00628)  
  
NeuralODEs are new class of deep learning architectures aimed towards robustness and better explainability of Deep learning models. These are continuous time Deep learning architectures, a generalization of their discrete models (ResNets).   

<!-- ## Basic Navigation:
* `models` directory contains the architectures of different models implemented
* `script` contains the bash scripts to run the model with different configurations(or hyperparameters)
* `dataset` contains the scripts for prepararing dataset and preprocessing of dataset
* `Example-Notebook` has pre-trained experiments and visualization of results
* `trainer.py` contains the code for training the model 
* `test.py` contains the code for testing the model and Inference mode -->

The aim of this project is to improve explainability of U-Net models using Continuous time models like NeuralODEs and visualizing the deep neural networks behavious across time steps
<br>
<br>

## Usage
* `config.json`: Contains the Hyperparameters used for the model training procedure.  
* Edit the `config.json` file for changing the model parameters and further experimentation. 
* `config.json` contains the following Hyperparameters: 
    * `batch_size`: The batch size to load the data
    * `lr` : Learning rate to use in training
    * `model` : Model Name ( available options: unet, neural_ode_convnet, neural_ode_unet)
    * `in_channel` : A number of images to use for input
    * `epochs`: The training epochs to run
    * `resume` : Model Trianing resume
    * `drop_rate` : Drop-out Rate
    * `data` : Label data type (Default = complete)
    * `img_root`: The directory containing the training image dataset
    * `label_root`: The directory containing the training label dataset
    * `output_root` : The directory containing the result predictions
    * `ckpt_root` : The directory containing the checkpoint files
    <br>
    <br>

### Installing Dependencies:
```bash
pip install requirements.txt
```
<br>
### Training the Model on BraTS Dataset:
```bash
python -m trainer.py
```
<br> 

## Dataset:

Multi-Modal MRI Dataset from BraTS 2020 Challenge  
`Dataset specs:`
* File : One file has a Multi-Modal MRI Data of one subject
* File Format: All files are .nii.gz files can be loaded using nibabel
* Image dimensions: 240(slice width) x 240(slice Height) x 155 (number of slices) x 4(Number of modalities i.e. T1, T2, FLAIR, T1ce)
* Labels:
    * `Ch 0`: Background
    * `Ch 1`: Necrotic and Non-Enhancing Tumor
    * `Ch 2`: Edema
    * `Ch 3`: Enhancing Tumor
<br>

## Models:
* U-Net model  
```bash
models/unet.py
```
<div align = "center">
    <img src = "https://i.imgur.com/OXtVFvT.png">
    <br>
    <br>
    <em align = "center">Fig Description: U-Net Diagram</em>
    <br>
</div>
<br>
<br>

* NeuralODE U-Net Model
```bash
models/neural_ode_unet.py
```
* NeuralODE ConvNet Model
```bash
models/neural_ode_convnet.py
```
<br>

## Folder Structure of Repository:

```
 ┣ checkpoint # contains the checkpoint files from the training process
 ┣ dataset
 ┃ ┣ dataset.py          ## used to load the preprocessed data
 ┃ ┣ preprocess.py       ## Preprocessing script for the dataset
 ┃ ┗ __init__.py
 ┣ models
 ┃ ┣ neural_ode_convnet.py             ## contains NeuralODE Convolutional Net Architecture code
 ┃ ┣ neural_ode_unet.py          ## contains NeuralODE U-Net Model Architecture code
 ┃ ┣ unet.py        ## contains U-Net Model architecture code
 ┃ ┗ __init__.py
 ┣ output
 ┣ config.py
 ┣ NeuralODE for Brain Tumor Segmentation on BraTS dataset.ipynb
 ┣ README.md
 ┣ test.py           ## Contains Test scripts
 ┣ trainer.py        ## used to train the model by running this script
 ┗ utils.py          ## contains helper Functions
 ┣ requirements.txt        ## Dependencies needed for training the model
 ┗ utils.py          ## contains helper Functions
 
```
<br>

## References:  
### BraTS Dataset References:
[1] [B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694](https://pubmed.ncbi.nlm.nih.gov/25494501/)   
[2] [S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117](https://pubmed.ncbi.nlm.nih.gov/28872634/)  
[3]  [S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)](https://arxiv.org/abs/1811.02629)

### Model Citations:  
[4] [A Neural Ordinary Differential Equation Model for Visualizing Deep Neural Network Behaviors in Multi-Parametric MRI based Glioma Segmentation](https://arxiv.org/abs/2203.00628) 
