import torch
import torch.nn as nn


class View(nn.Module):
  def __init__(self, shape):
    super(View, self).__init__()
    self.shape = shape

  def forward(self, x):
    return x.view(*self.shape)


class ConditionalInstanceNorm2d(nn.Module):
  def __init__(self, num_features, num_params):
    super().__init__()
    self.num_features = num_features
    self.InstNorm = nn.InstanceNorm2d(num_features, affine=False)
    self.affine = nn.Linear(num_params, num_features * 2)
    self.affine.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.affine.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, x, y):
    out = self.InstNorm(x)
    gamma, beta = self.affine(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out


class ConditionalInstanceNorm1d(nn.Module):
  def __init__(self, num_features, num_params):
    super().__init__()
    self.num_features = num_features
    self.InstNorm = nn.InstanceNorm1d(num_features, affine=False)
    self.affine = nn.Linear(num_params, num_features * 2)
    self.affine.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.affine.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, x, y):
    out = self.InstNorm(x)
    gamma, beta = self.affine(y).chunk(2, 1)
    out = gamma.view(-1, 1, self.num_features) * out + beta.view(-1, 1, self.num_features)
    return out



class ConditionalModule(nn.Module):
  def __init__(self):
    super(ConditionalModule, self).__init__()

  def forward(self, x, y):
    for name, mod in self.named_children():
      if 'CondInstNorm' in name:
        x = mod(x, y)
      else:
        x = mod(x)
    return x


class Generator(nn.Module):
  def __init__(self, params):
    super(Generator, self).__init__()
    nf = params.ng_filters
    self.x_size = params.x_size
    self.y_size = params.y_size

    out_channels = params.num_channels 
    layers = {}
    for ii in range(1, params.ng_layers+1):
      in_channels = out_channels
      if ii == params.ng_layers:
        out_channels = params.pred_interval - 1
      else:
        out_channels = params.ng_filters

      layers['Conv2d_%i'%ii] = nn.Conv2d(in_channels, out_channels, **params.conv_kwargs)
      if ii < params.ng_layers:
        if params.cond_BN:
          layers['CondInstNorm_%i'%ii] = ConditionalInstanceNorm2d(out_channels, params.num_params)
        else:
          layers['BatchNorm_%i'%ii] = nn.BatchNorm2d(out_channels, **params.batchnorm_kwargs)
        layers['LeakyReLU_%i'%ii] = nn.LeakyReLU(params.disc_LReLU_alpha, inplace=True)
      else:
        #layers['Last'] = nn.LeakyReLU(True)
        continue

    self.layers = layers
    self.main = ConditionalModule()
    [self.main.add_module(name, layers[name]) for name in layers.keys()]

  def forward(self, x, y):
    new = self.main(x,y)
    add_cum = torch.cumsum(new, axis=1) + x[:,1:,:,:] # new cumsum = cumsum(t+1...t+N) + cum(t)
    generated = torch.stack([new, add_cum], dim=1) # shape (bs, 2, 6, 124, 5)
    fullweek = torch.cat([torch.unsqueeze(x, dim=2), generated], dim=2) # shape (bs, 2, 7, 124, 5)
    return fullweek


class Discriminator(nn.Module):
  def __init__(self, params):
    super(Discriminator, self).__init__()

    layers = {}
    in_channels = params.num_channels*params.pred_interval
    out_channels = params.nd_filters
    x_map_size, y_map_size = [params.x_size, params.y_size]
    k = params.conv_kwargs['kernel_size']
    dk = params.downconv_kwargs['kernel_size']
    dsx, dsy = params.downconv_kwargs['stride']

    for ii in range(1, params.nd_layers+1):
      if ii==1:
        layers['Conv2d_%i'%ii] = nn.Conv2d(in_channels, out_channels, **params.conv_kwargs)
      else:
        layers['Conv2d_%i'%ii] = nn.Conv2d(in_channels, out_channels, **params.downconv_kwargs)
        x_map_size = int((x_map_size  - dk)/dsx + 1)
        y_map_size = int((y_map_size  - dk)/dsy + 1)

      if params.disc_batchnorm:
        if params.cond_BN:
          layers['CondInstNorm_%i'%ii] = ConditionalInstanceNorm2d(out_channels, params.num_params)
        else:
          layers['BatchNorm_%i'%ii] = nn.BatchNorm2d(out_channels, **params.batchnorm_kwargs)
      layers['LeakyReLU_%i'%ii] = nn.LeakyReLU(params.disc_LReLU_alpha, inplace=True)

      if params.disc_dropout:
        layers['Dropout2d_%i'%ii] = nn.Dropout2d(params.disc_dropout)

      if ii != params.nd_layers:
        in_channels = out_channels
      
    layers['Flat'] = nn.Flatten()
    layers['Linear'] = nn.Linear(out_channels*x_map_size*y_map_size, 1)

    self.layers = layers
    self.main = ConditionalModule()
    [self.main.add_module(name, layers[name]) for name in layers.keys()]

  def forward(self, x, y):
    return self.main(x,y)


def get_weights_function(params):
  def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
      if params['conv_init'] == 'truncated_normal':
        trunc_normal(m, std=params['conv_scale'])
      elif params['conv_init'] == 'normal':
        nn.init.normal_(m.weight.data, 0.0, params['conv_scale'])

      if params['conv_bias'] is not None:
        m.bias.data.fill_(params['conv_bias'])
    elif classname.find('Linear') != -1:
      if params['linear_scale'] is not None:
        nn.init.normal_(m.weight.data, 0.0, params['linear_scale'])
      if params['linear_bias'] is not None:
        m.bias.data.fill_(params['linear_bias'])
    elif classname.find('BatchNorm') != -1:
      if params['bn_bias'] is not None:
        m.bias.data.fill_(params['bn_bias'])
      if params['bn_weight'] is not None:
        m.weight.data.fill_(params['bn_weight'])

  def trunc_normal(m, std=0.02):
    nn.init.normal_(m.weight.data, 0.0, std)
    data = m.weight.data.detach().cpu().numpy()
    bad_vals = data[np.abs(data)>2*std]
    while len(bad_vals):
      data[np.abs(data)>2*std] = np.random.normal(loc=0., scale=std, size=data[np.abs(data)>2*std].shape)
      bad_vals = data[np.abs(data)>2*std]
    m.weight.data = torch.from_numpy(data).to(m.weight.data.device)

  return weights_init
