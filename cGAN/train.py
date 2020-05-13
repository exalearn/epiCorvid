import sys
import os
import random
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader import get_data_loader, get_vald_loader
from utils.plotting import plot_samples, compare_curves
from networks import dcgan

def get_labels(flip_labels_alpha, decay, epoch):

  if flip_labels_alpha:
    prob = np.random.uniform(1, 0, 1)[0]
    alpha = flip_labels_alpha/(1.+epoch) if decay else flip_labels_alpha
    real_label, fake_label = (0, 1) if prob < alpha else (1, 0)
  else:
    real_label, fake_label = (1, 0)

  return real_label, fake_label

def train(params, writer):

  data_loader = get_data_loader(params)
  vald_loader = get_vald_loader(params)

  device = torch.device("cuda" if (torch.cuda.is_available() and params.ngpu > 0) else "cpu")

  netG = dcgan.Generator(params).to(device)
  logging.info(netG)
  #summary(netG, (params.z_dim, params.num_params))

  netD = dcgan.Discriminator(params).to(device)
  logging.info(netD)
  #summary(netD, (params.num_channels, params.x_out_size, params.y_out_size))

  num_GPUs = torch.cuda.device_count()
  logging.info('Using %d GPUs'%num_GPUs)
  if num_GPUs > 1:
    device_list = list(range(num_GPUs))
    netG = nn.DataParallel(netG, device_list)
    netD = nn.DataParallel(netD, device_list)

  netG.apply(dcgan.get_weights_function(params.gen_init))
  netD.apply(dcgan.get_weights_function(params.disc_init))

  if params.optim=='Adam':
    optimizerD = optim.Adam(netD.parameters(), lr=params.lr, betas=(params.adam_beta1, params.adam_beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=params.lr, betas=(params.adam_beta1, params.adam_beta2))
    criterion = nn.BCEWithLogitsLoss()

  iters = 0
  startEpoch = 0

  checkpoint = None
  if params.load_checkpoint:
    logging.info("Loading checkpoint %s"%params.load_checkpoint)
    checkpoint = torch.load(params.load_checkpoint)
  elif os.path.exists(params.checkpoint_file):
    logging.info("Loading checkpoint %s"%params.checkpoint_file)
    checkpoint = torch.load(params.checkpoint_file)

  if checkpoint:
    netG.load_state_dict(checkpoint['G_state'])
    netD.load_state_dict(checkpoint['D_state'])

    if params.load_optimizers_from_checkpoint:
      iters = checkpoint['iters']
      startEpoch = checkpoint['epoch']
      if params.optim=='Adam':
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
  
  logging.info("Starting Training Loop...")
  for epoch in range(startEpoch, startEpoch+params.num_epochs):
    netG.train()
    netD.train()
    for i, data in enumerate(data_loader, 0):
      iters += 1
      real = data[0].to(device)
      pars = data[1].to(device)
      b_size = real.size(0)
      if params.optim=='Adam':
        real_label, fake_label = get_labels(params.flip_labels_alpha, params.decay_label_flipping, epoch)
        label = torch.full((b_size,), real_label, device=device)
        
        netD.zero_grad()
        output = netD(real, pars).view(-1)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        if params.resample_pars:
          pars = 2.*torch.rand((b_size, params.num_params), device=device) - 1.
        noise = torch.randn((b_size, 1, params.z_dim), device=device)
        fake = netG(noise, pars) 
        label.fill_(fake_label)
        output = netD(fake.detach(), pars).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake, pars).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
      
    logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                 % (epoch, startEpoch+params.num_epochs, i, len(data_loader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
          
    # Scalars
    writer.add_scalar('Loss/Disc', errD.item(), iters)
    writer.add_scalar('Loss/Gen', errG.item(), iters)
    writer.add_scalar('Disc_output/real', D_x, iters)
    writer.add_scalar('Disc_output/fake', D_G_z1, iters)

    # Plots
    netG.eval()
    reals = []
    fakes = []
    vpars = []
    for i, data in enumerate(vald_loader, 0):
        real = data[0].numpy()
        pars = data[1].to(device)
        biopars = data[2]
        with torch.no_grad():
          noise = torch.randn((b_size, 1, params.z_dim), device=device)
          fake = netG(noise, pars).detach().cpu().numpy()
          fakes.append(fake)
          vpars.append(biopars.detach().numpy())
          reals.append(real)
    reals = np.concatenate(reals, axis=0)
    fakes = np.concatenate(fakes, axis=0)
    vpars = np.concatenate(vpars, axis=0)
    fig = plot_samples(fakes[:1], vpars[:1], tag=params.tag)
    writer.add_figure('samples', fig, iters, close=True)
    fig, score, fig2 = compare_curves(reals, fakes, vpars, tag=params.tag, norm=params.norm)
    writer.add_figure('peak_curve', fig, iters, close=True)
    writer.add_figure('MAE_dist', fig2, iters, close=True)
    writer.add_scalar('peak_curve_MAE', score, iters)

    # Save checkpoint
    if params.optim=='Adam':
      torch.save({'epoch': epoch, 'iters': iters, 'G_state': netG.state_dict(),'D_state': netD.state_dict(),
                  'optimizerG_state_dict': optimizerG.state_dict(), 
                  'optimizerD_state_dict': optimizerD.state_dict()}, 
                  params.checkpoint_file)
  

if __name__ == '__main__':

  torch.backends.cudnn.benchmark=True
  if len(sys.argv) != 3:
    logging.error("Usage", sys.argv[0], "configuration_YAML_file", "configuration")
    exit()

  params = YParams(os.path.abspath(sys.argv[1]), sys.argv[2])
  if not os.path.exists(params.experiment_dir):
    os.makedirs(os.path.abspath(params.experiment_dir))

  logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(params.experiment_dir, 'out.log'))
  params.log()
  tboard_writer = SummaryWriter(log_dir=os.path.join(params.experiment_dir, 'logs/'))

  params.experiment_dir = os.path.abspath(params.experiment_dir)
  params.checkpoint_file = os.path.join(params.experiment_dir, 'checkpt.tar')

  if params.seed:
    random.seed(params.seed)
    torch.manual_seed(params.seed)

  train(params, tboard_writer)
  tboard_writer.flush()
  tboard_writer.close()
