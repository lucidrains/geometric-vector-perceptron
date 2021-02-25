# GVP - Point Cloud

Geometric Vector Perceptron applied to Point Clouds

## To install:

1. `git clone ${repo_url}`
2. install packages:
	* `sidechainnet`: https://github.com/jonathanking/sidechainnet#installation
	* joblib, tqdm, numpy, einops, ...
	* torch (was developed using 1.7.1)
	* torch geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
	* `cd this_repo_folder` + `pip install .` OR `pip install geometric-vector-perceptron` (but installing from PyPi is not recommended for now - not updated)
	* any other just run: `pip install package_name`
3. Try to run the notebooks (they should run, report errors if encountered)
    * `proto_dev_model.ipynb`: shows how to gather the data and train a simple model on it, then reconstruct original struct and calculate improvement. 


## Descritpion: 

1. encode a protein (3d) into some features (scalars and position vectors)
    * we encode both point features and edge features
2. train the model to predict the right point features back
3. reconstruct the 3d case to see the improvement 


## TO DO LIST:

See the issues tab?

## Contribute

PRs and ideas are welcome. Describe a list of the changes you've made and provide tests/examples if possible (they're not requiered, but surely helps understanding).
