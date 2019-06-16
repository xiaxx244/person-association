
from __future__ import print_function

import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import glob
import numpy as np
import argparse
import cv2
from PIL import Image
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self, block, layers, last_conv_stride=2):

    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(
      block, 512, layers[3], stride=last_conv_stride)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x


def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  for key, value in state_dict.items():
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict


def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.

  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
  return model


class Model(nn.Module):
  def __init__(self, last_conv_stride=2):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)

  def forward(self, x):
    # shape [N, C, H, W]
    x = self.base(x)
    x = F.avg_pool2d(x, x.size()[2:])
    # shape [N, C]
    x = x.view(x.size(0), -1)

    return x
def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist

def may_transfer_modules_optims(modules_and_or_optims, device_id=-1):
  """Transfer optimizers/modules to cpu or specified gpu.
  Args:
    modules_and_or_optims: A list, which members are either torch.nn.optimizer 
      or torch.nn.Module or None.
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for item in modules_and_or_optims:
    if isinstance(item, torch.optim.Optimizer):
      transfer_optim_state(item.state, device_id=device_id)
    elif isinstance(item, torch.nn.Module):
      if device_id == -1:
        item.cpu()
      else:
        item.cuda(device=device_id)
    elif item is not None:
      print('[Warning] Invalid type {}'.format(item.__class__.__name__))

def get_im_names(im_dir, pattern='*.jpg', return_np=True, return_path=False):
  """Get the image names in a dir. Optional to return numpy array, paths."""
  im_paths = glob.glob(osp.join(im_dir, pattern))
  im_names = [osp.basename(path) for path in im_paths]
  ret = im_paths if return_path else im_names
  if return_np:
    ret = np.array(ret)
  return ret

class TransferVarTensor(object):
  """Return a copy of the input Variable or Tensor on specified device."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, var_or_tensor):
    return var_or_tensor.cpu() if self.device_id == -1 \
      else var_or_tensor.cuda(self.device_id)


class TransferModulesOptims(object):
  """Transfer optimizers/modules to cpu or specified gpu."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, modules_and_or_optims):
    may_transfer_modules_optims(modules_and_or_optims, self.device_id)


def set_devices(sys_device_ids):
  """
  It sets some GPUs to be visible and returns some wrappers to transferring 
  Variables/Tensors and Modules/Optimizers.
  Args:
    sys_device_ids: a tuple; which GPUs to use
      e.g.  sys_device_ids = (), only use cpu
            sys_device_ids = (3,), use the 4th gpu
            sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
            sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
  Returns:
    TVT: a `TransferVarTensor` callable
    TMO: a `TransferModulesOptims` callable
  """
  # Set the CUDA_VISIBLE_DEVICES environment variable
  import os
  visible_devices = ''
  for i in sys_device_ids:
    visible_devices += '{}, '.format(i)
  os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
  # Return wrappers.
  # Models and user defined Variables/Tensors would be transferred to the
  # first device.
  device_id = 0 if len(sys_device_ids) > 0 else -1
  TVT = TransferVarTensor(device_id)
  TMO = TransferModulesOptims(device_id)
  return TVT, TMO


def load_state_dict(model, src_state_dict):
  """Copy parameters and buffers from `src_state_dict` into `model` and its 
  descendants. The `src_state_dict.keys()` NEED NOT exactly match 
  `model.state_dict().keys()`. For dict key mismatch, just
  skip it; for copying error, just output warnings and proceed.

  Arguments:
    model: A torch.nn.Module object. 
    src_state_dict (dict): A dict containing parameters and persistent buffers.
  Note:
    This is modified from torch.nn.modules.module.load_state_dict(), to make
    the warnings and errors more detailed.
  """
  from torch.nn import Parameter

  dest_state_dict = model.state_dict()
  for name, param in src_state_dict.items():
    if name not in dest_state_dict:
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    try:
      dest_state_dict[name].copy_(param)
    except Exception, msg:
      print("Warning: Error occurs when copying '{}': {}"
            .format(name, str(msg)))

  src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
  '''
  if len(src_missing) > 0:
    #print("Keys not found in source state_dict: ")
    for n in src_missing:
      print('\t', n)
  '''
  dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
  '''
  if len(dest_missing) > 0:
    print("Keys not found in destination state_dict: ")
    for n in dest_missing:
      print('\t', n)
   '''

class Config(object):
    def __init__(self):
        # gpu ids
        self.sys_device_ids = [] # for cpu
        # Image Processing
        self.resize_h_w = (256,128)
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        # The last block of ResNet has stride 2. We can set the stride to 1 so that
        # the spatial resolution before global pooling is doubled.
        self.last_conv_stride = 1

        # This only contains model weight
        self.model_weight_file = 'model_data/tri_loss/model_weight.pth'


def pre_process_im(im, cfg):
    """Pre-process image.
    `im` is a numpy array with shape [H, W, 3], e.g. the result of
    matplotlib.pyplot.imread(some_im_path), or
    numpy.asarray(PIL.Image.open(some_im_path))."""

    # Resize.
    im = cv2.resize(im, cfg.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    # scaled by 1/255.
    im = im / 255.

    # Subtract mean and scaled by std
    im = im - np.array(cfg.im_mean)
    im = im / np.array(cfg.im_std).astype(float)

    # shape [H, W, 3] -> [1, 3, H, W]
    im = im.transpose(2, 0, 1)[np.newaxis]
    return im


def feature_extraction():
    cfg = Config()

    TVT, TMO = set_devices(cfg.sys_device_ids)

    #########
    # Model #
    #########

    model = Model(last_conv_stride=cfg.last_conv_stride)
    # Set eval mode. Force all BN layers to use global mean and variance, also disable dropout.
    model.eval()
    # Transfer Model to Specified Device.
    TMO([model])

    #####################
    # Load Model Weight #
    #####################

    used_file = cfg.model_weight_file or cfg.ckpt_file
    loaded = torch.load(used_file, map_location=(lambda storage, loc: storage))
    if cfg.model_weight_file == '':
        loaded = loaded['state_dicts'][0]
    load_state_dict(model, loaded)

    ###################
    # Extract Feature #
    ###################

    im_dir ='model_data/query'
    im_paths = get_im_names(im_dir, pattern='*.jpg', return_path=True, return_np=False)

    all_feat = []
    for i, im_path in enumerate(im_paths):
        im = np.asarray(Image.open(im_path).convert('RGB'))
        im = pre_process_im(im, cfg)
        im = Variable(TVT(torch.from_numpy(im).float()), volatile=True)
        feat = model(im)
        feat = feat.data.cpu().numpy()
        all_feat.append(feat)
        #if (i + 1) % 100 == 0:
            #print('{}/{} images done'.format(i, len(im_paths)))
    #print(all_feat)
    return all_feat


def reid():
    all_feat=feature_extraction()
    #print (len(all_feat))
    #with measure_time('Computing distance...', verbose=True):
      # query-gallery distance
    q_g_dist = compute_dist(all_feat[0], all_feat[1], type='euclidean')
    #print('{:<30}'.format('Single Query:'), end='')
    if q_g_dist >1.0:
      return False
    else:
      return True
