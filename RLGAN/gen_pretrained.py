import sys
import os
import random
import numpy as np
import h5py

import torch
import torch.nn as nn

from utils.YParams import YParams
from networks import dcgan

bounds = {
            'R0': [0., 5.],
            'WFHcomp': [0., 1.],
            'WFHdays': [1., 365.],
           }

def normalized_params(lo, hi, param, batchsize, device):
  """Return normalized parameter vector for a range of values (3parA dataset scheme)"""
  if lo==hi:
    simlo, simhi = bounds[param]
    const = 2.*(lo - simlo)/(simhi - simlo) - 1.
    return torch.Tensor(const*np.ones((batchsize, 1), dtype=np.float32)).to(device)
  else:
    simlo, simhi = bounds[param]
    ulo, uhi = map(lambda x: 2.*(x - simlo)/(simhi - simlo) - 1., [lo, hi])
    return (uhi - ulo)*torch.rand((batchsize, 1), device=device) + ulo


def unnormalized(dat, param):
  """Rescale back to original space for a given parameter"""
  simlo, simhi = bounds[param]
  return (simhi - simlo)*(0.5*(dat + 1.)) + simlo


def generate(params, checkpt, outname, num=64, batch=64):
  device = torch.device("cuda" if (torch.cuda.is_available() and params.ngpu > 0) else "cpu")

  netG = dcgan.Generator(params).to(device)

  num_GPUs = torch.cuda.device_count()
  print('Using %d GPUs'%num_GPUs)
  if num_GPUs > 1:
    device_list = list(range(num_GPUs))
    netG = nn.DataParallel(netG, device_list)

  checkpoint = None
  print("Loading checkpoint %s"%checkpt)
  checkpoint = torch.load(checkpt)
  netG.load_state_dict(checkpoint['G_state'])
  netG.eval()

  print("Starting generation...")
  Niters = num//batch
  out = np.zeros((num, 5, 124, 365)).astype('ushort')
  pars_norm = np.zeros((num, params.num_params)).astype(np.float)
  pars_orig = np.zeros((num, params.num_params)).astype(np.float)
  for i in range(Niters):
    with torch.no_grad():
      pars_R0 = normalized_params(2., 3., param='R0', batchsize=batch, device=device)
      pars_WFHcomp = normalized_params(0., 0.9, param='WFHcomp', batchsize=batch, device=device)
      pars_WFHdays = normalized_params(45, 90, param='WFHdays', batchsize=batch, device=device)
      pars = torch.cat([pars_R0, pars_WFHcomp, pars_WFHdays], axis=-1)

      noise = torch.randn((batch, 1, params.z_dim), device=device)
      fake = netG(noise, pars).detach().cpu().numpy() 
      out[i*batch:(i+1)*batch,:,:,:] = np.round(fake).astype('ushort')
      upars = pars.detach().cpu().numpy()
      print(upars.shape)
      print(unnormalized(upars[:,0], 'R0').shape)
      pars_norm[i*batch:(i+1)*batch,:] = upars
      pars_orig[i*batch:(i+1)*batch,:] = np.stack([unnormalized(upars[:,0], 'R0'),
                                                   unnormalized(upars[:,1], 'WFHcomp'),
                                                   unnormalized(upars[:,2], 'WFHdays')], axis=-1)
  print("Output: shape %s, type %s, size %f MB"%(str(out.shape), str(out.dtype), out.nbytes/1e6))
  with h5py.File(outname, 'w') as f:
    f.create_dataset('symptomatic3D', data=out)
    f.create_dataset('parBio', data=pars_orig)
    f.create_dataset('uniBio', data=pars_norm)
  print("Saved output to %s"%(outname))

  

if __name__ == '__main__':

  torch.backends.cudnn.benchmark=True
  if len(sys.argv) != 5:
    print("Usage", sys.argv[0], "configuration_YAML_file", "configuration", "checkpoint", "outfile")
    exit()

  params = YParams(os.path.abspath(sys.argv[1]), sys.argv[2])
  N = 64
  bs = 64
  generate(params, sys.argv[3], sys.argv[4], num=N, batch=bs)
