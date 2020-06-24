import sys
import os
import random
import numpy as np
import argparse
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.YParams import YParams
from utils.data_loader import prep_for_inference
from networks import rlgan


def format_dict(state_dict):
  # Model is saved from a DataParallel implementation, so we need to
  # create a new OrderedDict that does not contain `module.` in keys
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    name = k[7:] # remove `module. 
    new_state_dict[name] = v
  return new_state_dict


def inference(params, args, bs=1):

  # Load simulation data, control parameters
  states = prep_for_inference(args.data, params)
  print(states.shape)

  # Load generator network
  device = torch.device("cuda" if (torch.cuda.is_available() and params.ngpu > 0) else "cpu")
  netG = rlgan.Generator(params).to(device)
  print("Loading weights %s"%args.saved_weights)
  checkpoint = torch.load(args.saved_weights)
  netG.load_state_dict(format_dict(checkpoint['G_state']))
  netG.eval()

  # Randomly sample a "seed" starting state from the dataset
  seed_idx = np.random.choice(np.arange(states.shape[0]), size=(bs,), replace=False)
  start_state = states[seed_idx,:,21,:,:] # shape = (bs, 2, 124, 5)

  # Iteratively predict
  current_state = torch.from_numpy(start_state).to(device)
  day = 21
  end_day = 350
  st_time = time.time()
  while day<end_day:
    # Sample new controls:
    # Compliances are from {0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} 
    new_ctrls = torch.randint(10, size=(bs,2), device=device)/10.

    # Predict
    print('Predicting from day %d, with controls %s'%(day, str(new_ctrls[0].cpu().numpy())))
    new_week = netG(current_state, new_ctrls) # shape = (bs, 2, 7, 124, 5)
    new_state = new_week[:,:,-1,:,:].detach()
    current_state = new_state
    day += 7
  print('Prediction took %f seconds'%(time.time()-st_time))
    

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--saved_weights", default='./weights.tar', type=str,
                      help='Path to saved model weights')
  parser.add_argument("--yaml_config", default='./config.yaml', type=str,
                      help='Path to YAML file sotring configs')
  parser.add_argument("--config", default='default', type=str,
                      help='Config tag')
  parser.add_argument("--data", default='./data.h5', type=str,
                      help='Path to data')
  args = parser.parse_args()

  params = YParams(os.path.abspath(args.yaml_config), args.config)

  inference(params, args)
