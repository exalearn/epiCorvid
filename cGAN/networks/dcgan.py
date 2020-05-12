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
    nz = int(params.z_dim)
    nf = params.ng_filters
    self.x_out = params.x_out_size
    self.y_out = params.y_out_size

    x_map_size, y_map_size = [int(params.x_out_size/int(2**params.ng_layers)), int(params.y_out_size/int(2**params.ng_layers))]
    out_channels = nf * int(2**(params.ng_layers -1))

    layers = {}
    layers['Linear'] = nn.Linear(nz, out_channels*x_map_size*y_map_size)
    if params.cond_BN:
      layers['CondInstNorm'] = ConditionalInstanceNorm1d(out_channels*x_map_size*y_map_size, params.num_params)
    else:
      layers['BatchNorm'] = nn.BatchNorm1d(out_channels*x_map_size*y_map_size, **params.batchnorm_kwargs)
    layers['View'] = View(shape=[-1, out_channels, x_map_size, y_map_size])
    layers['ReLU'] = nn.ReLU(True)

    for ii in range(1, params.ng_layers+1):
      in_channels = out_channels

      if ii == params.ng_layers:
        out_channels = params.num_channels
      else:
        out_channels = int(out_channels/2)

      layers['ConvTranspose2d_%i'%ii] = nn.ConvTranspose2d(in_channels, out_channels, **params.conv_transpose_kwargs)

      if ii < params.ng_layers:
        if params.cond_BN:
          layers['CondInstNorm_%i'%ii] = ConditionalInstanceNorm2d(out_channels, params.num_params)
        else:
          layers['BatchNorm_%i'%ii] = nn.BatchNorm2d(out_channels, **params.batchnorm_kwargs)
        layers['ReLU_%i'%ii] = nn.ReLU(True)
      else:
        layers['Last'] = nn.ReLU(True)

    self.layers = layers
    self.main = ConditionalModule()
    [self.main.add_module(name, layers[name]) for name in layers.keys()]

  def forward(self, x, y):
    uncropped = self.main(x, y)
    return uncropped[:,:,1:1+self.x_out,1:1+self.y_out] 


class Discriminator(nn.Module):
  def __init__(self, params):
    super(Discriminator, self).__init__()

    layers = {}
    in_channels = params.num_channels
    out_channels = params.nd_filters
    x_map_size, y_map_size = [params.x_out_size, params.y_out_size]
    k = params.conv_kwargs['kernel_size']
    p = params.conv_kwargs['padding']

    for ii in range(1, params.nd_layers+1):
      layers['Conv2d_%i'%ii] = nn.Conv2d(in_channels, out_channels, **params.conv_kwargs)
      x_map_size = int((x_map_size + 2*p - k)/2 + 1) # conv stride = 2
      y_map_size = int((y_map_size + 2*p - k)/2 + 1)

      if params.disc_batchnorm and ii != 1:
        if params.cond_BN:
          layers['CondInstNorm_%i'%ii] = ConditionalInstanceNorm2d(out_channels, params.num_params)
        else:
          layers['BatchNorm_%i'%ii] = nn.BatchNorm2d(out_channels, **params.batchnorm_kwargs)
      layers['LeakyReLU_%i'%ii] = nn.LeakyReLU(params.disc_LReLU_alpha, inplace=True)

      if params.disc_dropout:
        layers['Dropout2d_%i'%ii] = nn.Dropout2d(params.disc_dropout)

      if ii != params.nd_layers:
        in_channels = out_channels
        out_channels = int(out_channels*2)
      
    #layers['View'] = View(shape=[-1, out_channels*x_map_size*y_map_size])
    layers['Flat'] = nn.Flatten()
    layers['Linear'] = nn.Linear(out_channels*x_map_size*y_map_size, 1)

    self.main = nn.Sequential()
    [self.main.add_module(name, layers[name]) for name in layers.keys()]
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
