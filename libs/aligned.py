'''
code adapted from https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch
in order to make it suitable for any image and any dataset
model.weight is trained from Market1501 Dataset using AlignedReID method
'''
from __future__ import print_function
import cPickle as pickle
import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from PIL import Image
from collections import defaultdict
import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import cv2
ospj = osp.join
ospeu = osp.expanduser

import threading
import Queue as queue
import time
import torch.nn.init as init
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

  def __init__(self, block, layers):
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
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.

  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
  return model


class Model(nn.Module):
  def __init__(self, local_conv_out_channels=128, num_classes=None):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True)
    planes = 2048
    self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
    self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
    self.local_relu = nn.ReLU(inplace=True)

    if num_classes is not None:
      self.fc = nn.Linear(planes, num_classes)
      init.normal(self.fc.weight, std=0.001)
      init.constant(self.fc.bias, 0)

  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    global_feat = F.avg_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    global_feat = global_feat.view(global_feat.size(0), -1)
    # shape [N, C, H, 1]
    local_feat = torch.mean(feat, -1, keepdim=True)
    local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
    # shape [N, H, c]
    local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

    if hasattr(self, 'fc'):
      logits = self.fc(global_feat)
      return global_feat, local_feat, logits

    return global_feat, local_feat

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

def parse_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(im_name[:8])
  else:
    parsed = int(im_name[9:13])
  return parsed

class Counter(object):
  """A thread safe counter."""

  def __init__(self, val=0, max_val=0):
    self._value = val
    self.max_value = max_val
    self._lock = threading.Lock()

  def reset(self):
    with self._lock:
      self._value = 0

  def set_max_value(self, max_val):
    self.max_value = max_val

  def increment(self):
    with self._lock:
      if self._value < self.max_value:
        self._value += 1
        incremented = True
      else:
        incremented = False
      return incremented, self._value

  def get_value(self):
    with self._lock:
      return self._value


class Enqueuer(object):
  def __init__(self, get_element, num_elements, num_threads=1, queue_size=20):
    """
    Args:
      get_element: a function that takes a pointer and returns an element
      num_elements: total number of elements to put into the queue
      num_threads: num of parallel threads, >= 1
      queue_size: the maximum size of the queue. Set to some positive integer
        to save memory, otherwise, set to 0.
    """
    self.get_element = get_element
    assert num_threads > 0
    self.num_threads = num_threads
    self.queue_size = queue_size
    self.queue = queue.Queue(maxsize=queue_size)
    # The pointer shared by threads.
    self.ptr = Counter(max_val=num_elements)
    # The event to wake up threads, it's set at the beginning of an epoch.
    # It's cleared after an epoch is enqueued or when the states are reset.
    self.event = threading.Event()
    # To reset states.
    self.reset_event = threading.Event()
    # The event to terminate the threads.
    self.stop_event = threading.Event()
    self.threads = []
    for _ in range(num_threads):
      thread = threading.Thread(target=self.enqueue)
      # Set the thread in daemon mode, so that the main program ends normally.
      thread.daemon = True
      thread.start()
      self.threads.append(thread)

  def start_ep(self):
    """Start enqueuing an epoch."""
    self.event.set()

  def end_ep(self):
    """When all elements are enqueued, let threads sleep to save resources."""
    self.event.clear()
    self.ptr.reset()

  def reset(self):
    """Reset the threads, pointer and the queue to initial states. In common
    case, this will not be called."""
    self.reset_event.set()
    self.event.clear()
    # wait for threads to pause. This is not an absolutely safe way. The safer
    # way is to check some flag inside a thread, not implemented yet.
    time.sleep(5)
    self.reset_event.clear()
    self.ptr.reset()
    self.queue = queue.Queue(maxsize=self.queue_size)

  def set_num_elements(self, num_elements):
    """Reset the max number of elements."""
    self.reset()
    self.ptr.set_max_value(num_elements)

  def stop(self):
    """Wait for threads to terminate."""
    self.stop_event.set()
    for thread in self.threads:
      thread.join()

  def enqueue(self):
    while not self.stop_event.isSet():
      # If the enqueuing event is not set, the thread just waits.
      if not self.event.wait(0.5): continue
      # Increment the counter to claim that this element has been enqueued by
      # this thread.
      incremented, ptr = self.ptr.increment()
      if incremented:
        element = self.get_element(ptr - 1)
        # When enqueuing, keep an eye on the stop and reset signal.
        while not self.stop_event.isSet() and not self.reset_event.isSet():
          try:
            # This operation will wait at most `timeout` for a free slot in
            # the queue to be available.
            self.queue.put(element, timeout=0.5)
            break
          except:
            pass
      else:
        self.end_ep()
    print('Exiting thread {}!!!!!!!!'.format(threading.current_thread().name))

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
    except Exception:
      print("Warning: Error occurs when copying '{}': {}"
            .format(name, str(msg)))

  src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
  '''
  if len(src_missing) > 0:
    print("Keys not found in source state_dict: ")
    for n in src_missing:
      print('\t', n)

  dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
  if len(dest_missing) > 0:
    #print("Keys not found in destination state_dict: ")
    for n in dest_missing:
      print('\t', n)
'''

def shortest_dist(dist_mat):
  """Parallel version.
  Args:
    dist_mat: numpy array, available shape
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`
      1) scalar
      2) numpy array, with shape [N]
      3) numpy array with shape [*]
  """
  m, n = dist_mat.shape[:2]
  dist = np.zeros_like(dist_mat)
  for i in range(m):
    for j in range(n):
      if (i == 0) and (j == 0):
        dist[i, j] = dist_mat[i, j]
      elif (i == 0) and (j > 0):
        dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
      elif (i > 0) and (j == 0):
        dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
      else:
        dist[i, j] = \
          np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
          + dist_mat[i, j]
  # I ran into memory disaster when returning this reference! I still don't
  # know why.
  # dist = dist[-1, -1]
  dist = dist[-1, -1].copy()
  return dist


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


def parallel_local_dist(x, y):
  """Parallel version.
  Args:
    x: numpy array, with shape [M, m, d]
    y: numpy array, with shape [N, n, d]
  Returns:
    dist: numpy array, with shape [M, N]
  """
  M, m, d = x.shape
  N, n, d = y.shape
  x = x.reshape([M * m, d])
  y = y.reshape([N * n, d])
  # shape [M * m, N * n]
  dist_mat = compute_dist(x, y, type='euclidean')
  dist_mat = (np.exp(dist_mat) - 1.) / (np.exp(dist_mat) + 1.)
  # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
  dist_mat = dist_mat.reshape([M, m, N, n]).transpose([1, 3, 0, 2])
  # shape [M, N]
  dist_mat = shortest_dist(dist_mat)
  return dist_mat

def local_dist(x, y):
  if (x.ndim == 2) and (y.ndim == 2):
    return meta_local_dist(x, y)
  elif (x.ndim == 3) and (y.ndim == 3):
    return parallel_local_dist(x, y)
  else:
    raise NotImplementedError('Input shape not supported.')


def low_memory_matrix_op(
    func,
    x, y,
    x_split_axis, y_split_axis,
    x_num_splits, y_num_splits,
    verbose=False):
  """
  For matrix operation like multiplication, in order not to flood the memory 
  with huge data, split matrices into smaller parts (Divide and Conquer). 
  
  Note: 
    If still out of memory, increase `*_num_splits`.
  
  Args:
    func: a matrix function func(x, y) -> z with shape [M, N]
    x: numpy array, the dimension to split has length M
    y: numpy array, the dimension to split has length N
    x_split_axis: The axis to split x into parts
    y_split_axis: The axis to split y into parts
    x_num_splits: number of splits. 1 <= x_num_splits <= M
    y_num_splits: number of splits. 1 <= y_num_splits <= N
    verbose: whether to print the progress
    
  Returns:
    mat: numpy array, shape [M, N]
  """

  if verbose:
    import sys
    import time
    printed = False
    st = time.time()
    last_time = time.time()

  mat = [[] for _ in range(x_num_splits)]
  for i, part_x in enumerate(
      np.array_split(x, x_num_splits, axis=x_split_axis)):
    for j, part_y in enumerate(
        np.array_split(y, y_num_splits, axis=y_split_axis)):
      part_mat = func(part_x, part_y)
      mat[i].append(part_mat)

      if verbose:
        if not printed:
          printed = True
        else:
          # Clean the current line
          sys.stdout.write("\033[F\033[K")
        '''
        #print('Matrix part ({}, {}) / ({}, {}), +{:.2f}s, total {:.2f}s'
              .format(i + 1, j + 1, x_num_splits, y_num_splits,
                      time.time() - last_time, time.time() - st))
        '''
        last_time = time.time()
    mat[i] = np.concatenate(mat[i], axis=1)
  mat = np.concatenate(mat, axis=0)
  return mat
class Config(object):
  def __init__(self):
    # gpu ids
    self.sys_device_ids = [0]
    self.seed = None
    # Image Processing

    # Just for training set
    self.crop_prob = 0
    self.crop_ratio = 1
    self.resize_h_w = (256, 128)

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]
    self.dataset='market1501'
    self.ids_per_batch = 32
    self.ims_per_id = 4
    self.train_mirror_type = ['random', 'always', None][0]
    self.train_shuffle = True
    self.trainset_part='trainval'
    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_mirror_type = ['random', 'always', None][2]
    self.test_shuffle = False
    prng = np.random
    ###############
    # ReID Model  #
    ###############

    self.local_dist_own_hard_sample = False

    self.normalize_feature = False

    self.local_conv_out_channels = 128
    self.global_margin = 1
    self.local_margin = 0

    # Identification Loss weight
    self.id_loss_weight = 0

    # global loss weight
    self.g_loss_weight = 1.0
    # local loss weight
    self.l_loss_weight = 0.0

	
    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    self.model_weight_file = 'model_data/aligned_reid/model_weight.pth'


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    global_feat, local_feat = self.model(ims)[:2]
    global_feat = global_feat.data.cpu().numpy()
    local_feat = local_feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return global_feat, local_feat

def low_memory_local_dist(x, y):
    x_num_splits = int(len(x) / 200) + 1
    y_num_splits = int(len(y) / 200) + 1
    z = low_memory_matrix_op(
      local_dist, x, y, 0, 0, x_num_splits, y_num_splits, verbose=True)
    return z


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

def reid(im1_path,im2_path):
  cfg = Config()
  TVT, TMO = set_devices(cfg.sys_device_ids)
  #train_set = create_dataset(**cfg.train_set_kwargs)
  #########
  # Model #
  #########
  model = Model(local_conv_out_channels=cfg.local_conv_out_channels)
  # Model wrapper
  model_w = DataParallel(model)
  im1=cv2.imread(im1_path)
  im2=cv2.imread(im2_path)
  im1=pre_process_im(im1, cfg)
  im2=pre_process_im(im2, cfg)
  map_location = (lambda storage, loc: storage)
  sd = torch.load(cfg.model_weight_file, map_location=map_location)
  load_state_dict(model, sd)

  use_local_distance = (cfg.l_loss_weight > 0) and cfg.local_dist_own_hard_sample
  feat_func=ExtractFeature(model_w, TVT)
  global_feat_im1,local_feat_im1=feat_func(im1)
  global_feat_im2,local_feat_im2=feat_func(im2)

  ###################
  # Global Distance #
  ###################

  
  # query-gallery distance using global distance
  global_q_g_dist = compute_dist(global_feat_im1 , global_feat_im2, type='euclidean')

  ##################
  # Local Distance #
  ##################

  # query-gallery distance using local distance
  local_q_g_dist = low_memory_local_dist(
      local_feat_im1, local_feat_im2)
  
  #########################
  # Global+Local Distance #
  #########################

  global_local_q_g_dist = global_q_g_dist + local_q_g_dist
  #print(global_local_q_g_dist)
  ##Threshold may need to revise further!
  if global_local_q_g_dist<18:
      return True
  else:
      return False
