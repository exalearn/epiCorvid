import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import h5py
from torch import Tensor

                
def normalize(x, norm):
  if norm=='log':
    return np.log1p(x)
  else: 
    return x


def get_data_loader(params):
  with h5py.File(params.data_path, 'r') as f:
    arr = f['symptomatic3D'][:params.num_train,:,:,:].astype(np.float32)
    uniPars = f['uniBio'][:params.num_train,params.param_inds].astype(np.float32)
    bioPars = f['parBio'][:params.num_train,params.param_inds].astype(np.float32)
  dataset = SliceDataset(params, arr, uniPars, bioPars)

  loader = DataLoader(dataset,
                      batch_size=params.batch_size,
                      shuffle=True,
                      num_workers=params.num_data_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=True)
  return loader

def get_vald_loader(params):
  with h5py.File(params.data_path, 'r') as f:
    arr = f['symptomatic3D'][params.num_train:,:,:,:].astype(np.float32)
    uniPars = f['uniBio'][params.num_train:,params.param_inds].astype(np.float32)
    bioPars = f['parBio'][params.num_train:,params.param_inds].astype(np.float32)
  normed_arr = normalize(np.moveaxis(arr, 2, 1), params.norm)
  dataset = SliceDataset(params, normed_arr, uniPars, bioPars)
  loader = DataLoader(dataset,
                      batch_size=params.batch_size,
                      shuffle=True,
                      num_workers=params.num_data_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=True)
  states = build_states(arr)
  return loader, states, bioPars



def prep_for_inference(path, params):
  with h5py.File(params.data_path, 'r') as f:
    newsympt = f['symptomatic3D'][:params.num_train,:,:,:].astype(np.float32)
  cumsympt = np.cumsum(newsympt, axis=-1)
  states = np.stack([np.moveaxis(newsympt, -1, 1), 
                     np.moveaxis(cumsympt, -1, 1)], axis=1)
  return states 



def build_states(sympt):
  agg = np.sum(sympt, axis=(1,2))
  cumsympt = np.cumsum(agg, axis=-1)
  states = np.stack([agg, cumsympt], axis=-1)
  return states

class SliceDataset(Dataset):

  def __init__(self, params, sympt, uni, bio):
    self.arr = sympt
    self.uni = uni
    self.bio = bio
    self.len = sympt.shape[0]
    self.interval = params.pred_interval
    self.norm = params.norm
    self.start, self.end = params.start_end_weeks

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    t1 = 7*torch.randint(self.start, self.end, size=()).item()
    t2 = t1 + self.interval
    newsympt=self.arr[idx]
    uniPar=self.uni[idx]
    bioPar=self.bio[idx]

    cumsympt = np.cumsum(newsympt, axis=2)
    # Individual slices
    '''
    slice1 = np.moveaxis(np.concatenate([newsympt[:,:,t1:t1+1], cumsympt[:,:,t1:t1+1]], axis=-1), -1, 0)
    slice2 = np.moveaxis(np.concatenate([newsympt[:,:,t2:t2+1], cumsympt[:,:,t2:t2+1]], axis=-1), -1, 0)
    if self.norm=='log':
        slice1 = np.log1p(slice1)
        slice2 = np.log1p(slice2)
    return (torch.from_numpy(slice1), torch.from_numpy(slice2), 
           torch.Tensor(uniPar), torch.Tensor(bioPar))
    '''
    ts_newsympt = np.moveaxis(newsympt[:,:,t1:t2], -1, 0)
    ts_cumsympt = np.moveaxis(cumsympt[:,:,t1:t2], -1, 0)
    return torch.from_numpy(np.stack([ts_newsympt, ts_cumsympt], axis=0)), torch.Tensor(uniPar), torch.Tensor(bioPar)
