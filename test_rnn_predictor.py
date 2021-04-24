import sys
from release.rnn_predictor import RNNPredictor

sys.path.append('./OpenChem/')
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
predictor_tokens = tokens + [' ']
path_to_params = './checkpoints/logP/model_parameters.pkl'
path_to_checkpoint = './checkpoints/logP/fold_'
my_predictor = RNNPredictor(path_to_params, path_to_checkpoint, predictor_tokens)
canonical_smiles, prediction, invalid_smiles = my_predictor.predict('C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)CO)O)O)N')
print('canonical', canonical_smiles)
print('prediction', prediction)
print('invalid', invalid_smiles)