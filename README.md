# Tallas Protoype
A prototype model for RNAi prediction in the aid of drug discovery.

## Predictor
An ML model to predict activity of RNAi molecules.

### Run
Make sure you have enabled the tallas conda environemnt before running the code. From the predictor directory run the following code `python predictor.py`.

## Generator
An ML model to generate RNAi molecules

## Prototype
An RL model that combines the generator and predictor to facilitate pseudo-unsupervised RNAi selection.


### Conda Environment
#### Installation
Create a local conda environment with all the necessary dependencies using the following command `conda env create -f=conda.yml`. The environment name is tallas

#### Saving
Save any environmental changes using the following command `conda env export -n tallas | grep -v "^prefix: " > conda.yml`.

