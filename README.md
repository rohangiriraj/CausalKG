# Introduction

This is the official implementation of the paper "A hybrid method for Causal Inference in Knowledge Graphs". We use many existing open source libraries for implementing the method
proposed in the paper. Please follow the setup instructions to install and get the code running.

# Setup and Installation

For running the experiments, we created a new environment using Conda, with python version 3.8.5.
You can create a new environment in conda using 
`conda create -n <env-name> python=3.8`

Once created, you can enter the new environment by `conda activate <env-name>`

You can then install all the required dependencies by running this command:
`pip install -r requirements.txt`
## Training custom embeddings
If you want to train your own tucker embeddings with custom hyperparameters follow these steps, else run the hybrid algorithm.
1. Head over to [pykg2vec](https://github.com/Sujit-O/pykg2vec) and follow the instructions to install the pykg2vec package for training custom embeddings.
2. `cd` into the examples folder of the cloned pykg2vec repository.
3.  To run the pykg2vec embedding with the same hyperparameters run the command 
`python train.py  -exp True -mn TuckER -ds freebase15k_237 -hpf custom_hp.yaml`
 
 in the examples folder. This creates the embeddings and stores them in the `/datasets/dataset-name/embeddings` folder as `.tsv` files.  You might have to include the full path to the `custom_hp.yaml` file included in this repo.
 4. Depending on the version of `pykg2vec` you may have to add additional details to the `.yaml` file located in the `site-packages` of the conda environment (installation location for `pykg2vec`).
5. Once the custom embeddings are trained, you can follow the next steps to execute the algorithm for causal discovery.

## Running the hybrid algorithm
1. Before running the project, check if the required embedding (if custom training is done) are located in the same folder as that of the script.
1. You can run the hybrid algorithm by running the following command in the project folder where `hybrid.py` is located.
`python hybrid.py -dataset fb15k-237 -algorithm DirectLiNGAM -plot True`
2. The output of the above command will be a text file `results_hybrid.txt`	which contains the execution time, the mean p-value and the causal order.
It also plots the Directed Acyclic Graph of the causal order output by the algorithm. 

