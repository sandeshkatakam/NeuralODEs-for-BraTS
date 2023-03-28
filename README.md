# NeuralODEs for Brain Tumor Segmentation

Implementation of "A Neural Ordinary Differential Equation Model for Visualizing Deep Neural Network Behaviors in Multi-Parametric MRI based Glioma Segmentation"  
[Link to the paper](https://arxiv.org/abs/2203.00628)  
  
NeuralODEs are new class of deep learning architectures aimed towards robustness and better explainability of Deep learning models. These are continuous time Deep learning architectures, a generalization of their discrete models (ResNets).   

## Basic Navigation:
* `models` directory contains the architectures of different models implemented
* `script` contains the bash scripts to run the model with different configurations(or hyperparameters)
* `dataset` contains the scripts for prepararing dataset and preprocessing of dataset
* `Example-Notebook` has pre-trained experiments and visualization of results
* `trainer.py` contains the code for training the model 
* `test.py` contains the code for testing the model and Inference mode
* ``

The aim of this project is to improve explainability of U-Net models using Continuous time models like NeuralODEs and visualizing the deep neural networks behavious across time steps
