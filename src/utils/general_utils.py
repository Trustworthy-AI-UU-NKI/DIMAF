import random
import torch
import os
import pickle
import json

import numpy as np

def set_seed(seed):
    """ Set seed for reproducible experiments. """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_pkl(dir, name, dict):
	""" Save dict to pickle file. """
	os.makedirs(dir, exist_ok=True)
	filename = os.path.join(dir, name)
	with open(filename,'wb') as f:
	    pickle.dump(dict, f)

def load_pkl(dir, name):
	""" Load pickle file to dict. """
	filename = os.path.join(dir, name)
	with open(filename, 'rb') as f:
		file = pickle.load(f)
	return file

def save_json(dir, name, dict):
	""" Save dict to json file. """
	os.makedirs(dir, exist_ok=True)
	filename = os.path.join(dir, name)
	with open(filename, 'w') as json_file:
		json.dump(dict, json_file, indent=4)

def save_exp_settings(args):
	""" Save settings of an experiments using the specified arguments. """
	os.makedirs(args.result_dir, exist_ok=True)

	# Save the args namespace to a JSON file
	args_dict = vars(args)  # Convert Namespace to dictionary
	json_path = os.path.join(args.result_dir, 'args.json')

	with open(json_path, 'w') as json_file:
		json.dump(args_dict, json_file, indent=4)

