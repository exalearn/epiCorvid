import sys
import os
import random
import numpy as np
import time

import torch
import torch.nn as nn

from utils.YParams import YParams
from networks import dcgan


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
  for i in range(Niters):
    with torch.no_grad():
      if '2parB' in params.data_path:
        # Different normalized range for different dataset
        pars_R0 = 1.4*torch.rand((batch, 1), device=device) - 0.4
        pars_TriggerDay = 1.3791*torch.rand((batch, 1), device=device) - 1.
        pars = torch.cat([pars_R0, pars_TriggerDay], axis=-1)
      elif '3parA' in params.tag:
        pars_R0 = 0.4*torch.rand((batch, 1), device=device) - 0.2
        pars_WFHcomp = 1.8*torch.rand((batch, 1), device=device) - 1.
        pars_WFHdays = 0.24725*torch.rand((batch, 1), device=device) - 0.75275
        pars = torch.cat([pars_R0, pars_WFHcomp, pars_WFHdays], axis=-1)
      else:
        pars = 2.*torch.rand((batch, 2), device=device) - 1.
      noise = torch.randn((batch, 1, params.z_dim), device=device)
      fake = netG(noise, pars).detach().cpu().numpy() 
      if params.norm == 'log':
        fake = np.exp(fake) - 1.
      out[i*batch:(i+1)*batch,:,:,:] = np.round(fake).astype('ushort')
      pars_norm[i*batch:(i+1)*batch,:] = pars.detach().cpu().numpy()
  print("Output: shape %s, type %s, size %f MB"%(str(out.shape), str(out.dtype), out.nbytes/1e6))
  np.save(outname+'_dat.npy', out)
  np.save(outname+'_par.npy', pars_norm)
  print("Saved output to %s"%(outname+'_dat.npy'))
  print("Saved parameters to %s"%(outname+'_par.npy'))

  

if __name__ == '__main__':

  torch.backends.cudnn.benchmark=True
  if len(sys.argv) != 5:
    print("Usage", sys.argv[0], "configuration_YAML_file", "configuration", "checkpoint", "outfile")
    exit()

  params = YParams(os.path.abspath(sys.argv[1]), sys.argv[2])
  N = 32768
  bs = 8192
  start = time.time()
  generate(params, sys.argv[3], sys.argv[4], num=N, batch=bs)
  end = time.time()
  print("Generating %d with batchsize %d took %f s"%(N,bs,end-start))


