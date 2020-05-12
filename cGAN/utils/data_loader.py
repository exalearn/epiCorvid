import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import h5py
from torch import Tensor

num_train = 8500


def normalize(x, norm):
  if norm=='log':
    return np.log1p(x)
  else: 
    return x


def get_data_loader(params):
  with h5py.File(params.data_path, 'r') as f:
    arr = f['symptomatic3D'][:num_train,:,:,:].astype(np.float32)
    uniPars = f['uniBio'][:num_train,:params.num_params].astype(np.float32)
  arr = normalize(np.moveaxis(arr, 2, 1), params.norm)
  dataset = TensorDataset(Tensor(arr), Tensor(uniPars))

  loader = DataLoader(dataset,
                      batch_size=params.batch_size,
                      shuffle=True,
                      num_workers=params.num_data_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=True)
  return loader

def get_vald_loader(params):
  with h5py.File(params.data_path, 'r') as f:
    arr = f['symptomatic3D'][num_train:,:,:,:].astype(np.float32)
    uniPars = f['uniBio'][num_train:,:params.num_params].astype(np.float32)
  arr = normalize(np.moveaxis(arr, 2, 1), params.norm)
  dataset = TensorDataset(Tensor(arr), Tensor(uniPars))

  loader = DataLoader(dataset,
                      batch_size=params.batch_size,
                      shuffle=True,
                      num_workers=params.num_data_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=True)
  return loader

