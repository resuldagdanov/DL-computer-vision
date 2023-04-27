"""
  Implements convolutional networks in PyTorch.
  WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import random
from eecs598 import Solver
from a3_helper import svm_loss, softmax_loss
from fully_connected_networks import *


def hello_convolutional_networks():
  """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
  """
  print('Hello from convolutional_networks.py!')


class Conv(object):

  @staticmethod
  def forward(x: torch.Tensor,
              w: torch.Tensor,
              b: torch.Tensor,
              conv_param: dict):
    """
      A naive implementation of the forward pass for a convolutional layer.
      The input consists of N data points, each with C channels, height H and
      width W. We convolve each input with F different filters, where each filter
      spans all C channels and has height HH and width WW.

      Input:
      - x: Input data of shape (N, C, H, W)
      - w: Filter weights of shape (F, C, HH, WW)
      - b: Biases, of shape (F,)
      - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input. 
        
      During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
      along the height and width axes of the input. Be careful not to modfiy the original
      input x directly.

      Returns a tuple of:
      - out: Output data, of shape (N, F, H', W') where H' and W' are given by
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
      - cache: (x, w, b, conv_param)
    """

    # Extract the hyperparameters
    stride = conv_param['stride']
    pad = conv_param['pad']

    # Pad the input
    x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad)).to(x.dtype).to(x.device)

    # Extract the shapes
    N, C, H, W = x.shape
    CC, _, HH, WW = w.shape

    # Compute the output shape
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Allocate memory for the output
    out = torch.zeros((N, CC, H_out, W_out)).to(x.dtype).to(x.device)

    # Perform the convolution
    for n in range(N):
      for c in range(CC):

        for i in range(H_out):
          for j in range(W_out):

            # Compute the convolution
            out[n, c, j, i] = (x_padded[n, :, j * stride : j * stride + HH, i * stride : i * stride + WW] * w[c, :, :, :]).sum() + b[c]

    cache = (x, w, b, conv_param)

    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    """
      A naive implementation of the backward pass for a convolutional layer.

      Inputs:
      - dout: Upstream derivatives.
      - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

      Returns a tuple of:
      - dx: Gradient with respect to x
      - dw: Gradient with respect to w
      - db: Gradient with respect to b
    """

    # Extract batch, channel, height, width sizes
    x, w, b, conv_param = cache
    N_batch, C, H, W = x.shape
    C_channel, _, height, width = w.shape

    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    # Apply padding
    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad)).to(x.dtype).to(x.device)
    H_size = 1 + (H + 2 * pad - height) // stride
    W_size = 1 + (W + 2 * pad - width) // stride

    # Construct the output
    dx_pad = torch.zeros_like(x_pad).to(x.dtype).to(x.device)
    dx = torch.zeros_like(x).to(x.dtype).to(x.device)
    dw = torch.zeros_like(w).to(x.dtype).to(x.device)
    db = torch.zeros_like(b).to(x.dtype).to(x.device)

    # Naively writing for loop for each batch, channel size, height, and width dimensions
    for n in range(N_batch):

      for c in range(C_channel):
        db[c] += torch.sum(dout[n, c])

        for h_ in range(0, H_size):
          for w_ in range(0, W_size):

            dw[c] += x_pad[n, :, h_ * stride : h_ * stride + height, w_ * stride : w_ * stride + width] * dout[n, c, h_, w_]
            dx_pad[n, :, h_ * stride : h_ * stride + height, w_ * stride : w_ * stride + width] += w[c] * dout[n, c, h_, w_]
    
    # Extract dx from dx_pad
    dx = dx_pad[:, :, pad : pad + H, pad : pad + W]
    
    return dx, dw, db


class MaxPool(object):

  @staticmethod
  def forward(x: torch.Tensor,
              pool_param: dict):
    """
      A naive implementation of the forward pass for a max-pooling layer.

      Inputs:
      - x: Input data, of shape (N, C, H, W)
      - pool_param: dictionary with the following keys:
        - 'pool_height': The height of each pooling region
        - 'pool_width': The width of each pooling region
        - 'stride': The distance between adjacent pooling regions
      No padding is necessary here.

      Returns a tuple of:
      - out: Output data, of shape (N, C, H', W') where H' and W' are given by
        H' = 1 + (H - pool_height) / stride
        W' = 1 + (W - pool_width) / stride
      - cache: (x, pool_param)
    """

    # Extract batch, channel, height, width sizes
    N, C, H, W = x.shape

    # Hyperparameters
    height = pool_param.get('pool_height', 2)
    width = pool_param.get('pool_width', 2)
    
    stride = pool_param.get('stride', 2)

    assert (H - height) % stride == 0, 'Sanity Check Status: Max Pool Failed in Height'
    assert (W - width) % stride == 0, 'Sanity Check Status: Max Pool Failed in Width'

    H_prime = 1 + (H - height) // stride
    W_prime = 1 + (W - width) // stride
    
    out = torch.zeros((N, C, H_prime, W_prime)).to(x.device).to(x.dtype)
    
    # Naive looping with for loops in height and width
    for h in range(H_prime):
      for w in range(W_prime):

        out[:, :, h, w] = torch.max(torch.max(x[:,
                                                :,
                                                h*stride : h*stride+height,
                                                w*stride : w*stride+width], -1)[0], -1)[0]

    cache = (x, pool_param)

    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    """
      A naive implementation of the backward pass for a max-pooling layer.
      Inputs:
      - dout: Upstream derivatives
      - cache: A tuple of (x, pool_param) as in the forward pass.
      Returns:
      - dx: Gradient with respect to x
    """

    # Extract constants
    x, pool_param = cache

    # Extract batch, channel, height, width sizes
    N, C, H, W = x.shape

    # Hyperparameters
    height = pool_param.get('pool_height', 2)
    width = pool_param.get('pool_width', 2)

    stride = pool_param.get('stride', 2)

    H_prime = 1 + (H - height) // stride
    W_prime = 1 + (W - width) // stride

    # Construct output
    dx = torch.zeros_like(x).to(x.device)

    # Naively writing for loop for each batch, channel size, height, and width dimensions
    for n in range(N):
      
      # Loop through channels
      for c in range(C):
      
        # Loop through height, and width dimensions
        for h in range(H_prime):
          for w in range(W_prime):

            # The max-pooling backward pass
            ind = torch.argmax(x[n, c, h*stride : h*stride+height, w*stride : w*stride+width])
            ind1, ind2 = ind // height, ind%width
            dx[n, c, h*stride : h*stride+height, w*stride : w*stride+width][ind1, ind2] = dout[n, c, h, w]
  
    return dx


class ThreeLayerConvNet(object):
  """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
  """

  def __init__(self,
               input_dims: tuple=(3, 32, 32),
               num_filters: int=32,
               filter_size: int=7,
               hidden_dim: int=100,
               num_classes: int=10,
               weight_scale: float=1e-3,
               reg: float=0.0,
               dtype=torch.float,
               device: str='cpu'):
    """
      Initialize a new network.
      Inputs:
      - input_dims: Tuple (C, H, W) giving size of input data
      - num_filters: Number of filters to use in the convolutional layer
      - filter_size: Width/height of filters to use in the convolutional layer
      - hidden_dim: Number of units to use in the fully-connected hidden layer
      - num_classes: Number of scores to produce from the final linear layer.
      - weight_scale: Scalar giving standard deviation for random initialization
        of weights.
      - reg: Scalar giving L2 regularization strength
      - dtype: A torch data type object; all computations will be performed using
        this datatype. float is faster but less accurate, so you should use
        double for numeric gradient checking.
      - device: device to use for computation. 'cpu' or 'cuda'
    """

    # All weights and biases should be stored in the dictionary self.params
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    # Dimensions: channel, height, width
    C, H, W = input_dims

    # Randomly initialize weight and bias parameters
    # Weights should be initialized from a Gaussian centered at 0.0
    self.params['W1'] = weight_scale * torch.randn(num_filters, C, filter_size, filter_size).to(dtype).to(device)
    self.params['b1'] = torch.zeros(num_filters).to(dtype).to(device)

    # Assuming a shape identical to the input image for the conv layer output
    self.params['W2'] = weight_scale * torch.randn(num_filters * H * W // 4, hidden_dim).to(dtype).to(device)
    self.params['b2'] = torch.zeros(hidden_dim).to(dtype).to(device)
    self.params['W3'] = weight_scale * torch.randn(hidden_dim, num_classes).to(dtype).to(device)
    self.params['b3'] = torch.zeros(num_classes).to(dtype).to(device)

  def save(self,
           path: str):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self,
           path: str):
    checkpoint = torch.load(path, map_location='cpu')
    
    self.params = checkpoint['params']
    self.dtype = checkpoint['dtype']
    self.reg = checkpoint['reg']
    print("load checkpoint file: {}".format(path))

  def loss(self,
           X: torch.Tensor,
           y: torch.Tensor=None):
    """
      Evaluate loss and gradient for the three-layer convolutional network.
      Input / output: Same API as TwoLayerNet.
    """

    X = X.to(self.dtype)

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # Pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = W1.shape[2]
    conv_param = {'stride': 1,
                  'pad': (filter_size - 1) // 2}

    # Pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2,
                  'pool_width': 2,
                  'stride': 2}

    # The forward pass for the three-layer convolutional net
    out1, cache1 = Conv_ReLU_Pool.forward(x=X,
                                          w=self.params['W1'],
                                          b=self.params['b1'],
                                          conv_param=conv_param,
                                          pool_param=pool_param)
    out2, cache2 = Linear_ReLU.forward(x=out1,
                                       w=self.params['W2'],
                                       b=self.params['b2'])
    out3, cache3 = Linear.forward(x=out2,
                                  w=self.params['W3'],
                                  b=self.params['b3'])
    scores = out3

    if y is None:
      return scores

    loss, grads = 0.0, {}

    # The backward pass for the three-layer convolutional net
    reg=torch.tensor(self.reg).to(X.device).to(X.dtype)
    loss, dout = softmax_loss(out3, y)
    
    loss += reg * torch.sum(torch.tensor([torch.sum(self.params['W%d' % i] ** 2) for i in [1, 2, 3]]))

    # Add L2 regularization for each weight layer
    dout, grads['W3'], grads['b3'] = Linear.backward(dout=dout,
                                                     cache=cache3)
    grads['W3'] += 2 * reg * self.params['W3']

    dout, grads['W2'], grads['b2'] = Linear_ReLU.backward(dout=dout,
                                                          cache=cache2)
    grads['W2'] += 2 * reg * self.params['W2']

    _, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dout=dout,
                                                          cache=cache1)
    grads['W1'] += 2 * reg * self.params['W1']

    return loss, grads


class DeepConvNet(object):
  """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and 
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:
    
    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
  """

  def __init__(self,
               input_dims: tuple=(3, 32, 32),
               num_filters: list=[8, 8, 8, 8, 8],
               max_pools: list=[0, 1, 2, 3, 4],
               batchnorm: bool=False,
               num_classes: int=10,
               weight_scale: float=1e-3,
               reg: float=0.0,
               weight_initializer: float=None,
               dtype=torch.float,
               device: str='cpu'):
    """
      Initialize a new network.

      Inputs:
      - input_dims: Tuple (C, H, W) giving size of input data
      - num_filters: List of length (L - 1) giving the number of convolutional
        filters to use in each macro layer.
      - max_pools: List of integers giving the indices of the macro layers that
        should have max pooling (zero-indexed).
      - batchnorm: Whether to include batch normalization in each macro layer
      - num_classes: Number of scores to produce from the final linear layer.
      - weight_scale: Scalar giving standard deviation for random initialization
        of weights, or the string "kaiming" to use Kaiming initialization instead
      - reg: Scalar giving L2 regularization strength. L2 regularization should
        only be applied to convolutional and fully-connected weight matrices;
        it should not be applied to biases or to batchnorm scale and shifts.
      - dtype: A torch data type object; all computations will be performed using
        this datatype. float is faster but less accurate, so you should use
        double for numeric gradient checking.
      - device: device to use for computation. 'cpu' or 'cuda'    
    """

    self.params = {}
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
  
    if device == 'cuda':
      device = 'cuda:0'
    
    # Sizes: channeli height, weight
    C, H, W = input_dims

    layers_dims = num_filters + [num_classes]

    # Initialize the parameters for the DeepConvNet
    for i in range(len(num_filters)):

      # Weights for conv and fully-connected layers should be initialized according to weight_scale
      if i==0:
        if weight_scale == "kaiming":
          self.params['W'+str(i+1)] = kaiming_initializer(Din=layers_dims[i],
                                                          Dout=C,
                                                          K=3, 
                                                          relu=True,
                                                          device=device,
                                                          dtype=dtype)
        else:
          self.params['W'+str(i+1)] = weight_scale * torch.randn(layers_dims[i],
                                                                 C, 3, 3).to(dtype).to(device)
        self.params['b'+str(i+1)] = torch.zeros(layers_dims[i]).to(dtype).to(device)
      
      else:
        if weight_scale == "kaiming":
          self.params['W'+str(i+1)] = kaiming_initializer(Din=layers_dims[i],
                                                          Dout=layers_dims[i-1],
                                                          K=3, 
                                                          relu=True,
                                                          device=device,
                                                          dtype=dtype)
        else:
          self.params['W'+str(i+1)] = weight_scale * torch.randn(layers_dims[i],
                                                                 layers_dims[i-1], 3, 3).to(dtype).to(device)
        self.params['b'+str(i+1)] = torch.zeros(layers_dims[i]).to(dtype).to(device)

    # Batchnorm scale (gamma) and shift (beta) parameters should be initilized to ones and zeros respectively
    if self.batchnorm == True:
        for i in range(len(num_filters)):
            self.params['gamma'+str(i+1)] = torch.ones(layers_dims[i]).to(dtype).to(device)
            self.params['beta' +str(i+1)] = torch.zeros(layers_dims[i]).to(dtype).to(device)
    
    i += 1
    
    downsample_rate = 2**(len(max_pools))

    if weight_scale == "kaiming":
      self.params['W'+str(i+1)] = kaiming_initializer(Din=num_filters[i-1] * H * W // downsample_rate**2,
                                                      Dout=layers_dims[i],
                                                      K=None, 
                                                      relu=True,
                                                      device=device,
                                                      dtype=dtype)
    else:
      self.params['W'+str(i+1)] = weight_scale * torch.randn(num_filters[i-1] * H * W // downsample_rate**2 ,
                                                             layers_dims[i]).to(dtype).to(device)
    self.params['b'+str(i+1)] = torch.zeros(layers_dims[i]).to(dtype).to(device)

    # With batch normalization we need to keep track of running means and variances
    self.bn_params = []

    if self.batchnorm:
      self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
    
    if not self.batchnorm:
      # Because of weight and bias
      params_per_macro_layer = 2
    
    else:
      # Because of weight, bias, gamma and beta
      params_per_macro_layer = 4
    
    num_params = params_per_macro_layer * len(num_filters) + 2

    msg = 'self.params has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
      msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
      assert param.device == torch.device(device), msg

      msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
      assert param.dtype == dtype, msg

  def save(self,
           path: str):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'max_pools': self.max_pools,
      'batchnorm': self.batchnorm,
      'bn_params': self.bn_params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self,
           path: str,
           dtype,
           device: str):
    checkpoint = torch.load(path, map_location='cpu')
    
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']

    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
      for p in ["running_mean", "running_var"]:
        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))

  def loss(self,
           X: torch.Tensor,
           y: torch.Tensor=None):
    """
      Evaluate loss and gradient for the deep convolutional network.
      Input / output: Same API as ThreeLayerConvNet.
    """

    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they behave differently during training and testing
    if self.batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    # Pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1,
                  'pad': (filter_size - 1) // 2}

    # Pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2,
                  'pool_width': 2,
                  'stride': 2}
    
    # Regularization
    reg = torch.tensor(self.reg).to(X.dtype).to(X.device)
    L2reg = 0

    scores = X
    cache_history = []
    
    # Implement the forward pass for the DeepConvNet
    for i in range(self.num_layers-1):
      
      # Use the fast versions of convolution and max pooling layers
      scores, cache = FastConv.forward(x=scores,
                                       w=self.params['W%d' % (i + 1)], 
                                       b=self.params['b%d' % (i + 1)],
                                       conv_param=conv_param)
      cache_history.append(cache)
      
      # Add regularization
      L2reg += torch.sum(self.params['W%d' % (i + 1)] ** 2)
      
      # Apply batch normalization operation
      if self.batchnorm:
        scores, cache = SpatialBatchNorm.forward(x=scores,
                                                 gamma=self.params['gamma%d' % (i + 1)],
                                                 beta=self.params['beta%d' % (i + 1)],
                                                 bn_param=self.bn_params[i])
        cache_history.append(cache)
      

      scores, cache = ReLU.forward(x=scores)
      cache_history.append(cache)
      
      # Apply max-pooling operation
      if i in self.max_pools:
        scores, cache = FastMaxPool.forward(x=scores,
                                            pool_param=pool_param)
        cache_history.append(cache)
    
    i += 1

    scores, cache = Linear.forward(x=scores,
                                   w=self.params['W%d' % (i + 1)],
                                   b=self.params['b%d' % (i + 1)])
    cache_history.append(cache)

    # Add regularization
    L2reg += torch.sum(self.params['W%d' % (i + 1)] ** 2)
    L2reg *= self.reg

    if y is None:
      return scores

    loss, grads = 0.0, {}

    # Compute data loss using softmax and add L2 regularization
    loss, dout = softmax_loss(scores, y)
    loss += L2reg

    dout, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)] = Linear.backward(dout=dout,
                                                                           cache=cache_history.pop())
    grads['W%d' % (i + 1)] += 2 * reg * self.params['W%d' % (i + 1)]
    
    i -= 1

    # Implement the backward pass for the DeepConvNet
    while i >= 0:
      
      # Backward pass gradient of the max-pooling operation
      if i in self.max_pools:
        dout = FastMaxPool.backward(dout=dout,
                                    cache=cache_history.pop())
      
      # Backward pass gradient of the ReLU operation
      dout = ReLU.backward(dout=dout,
                           cache=cache_history.pop())

      # Backward pass gradient of the batch normalization operation
      if self.batchnorm:
        dout, grads['gamma%d' % (i + 1)], grads['beta%d' % (i + 1)] = SpatialBatchNorm.backward(dout=dout,
                                                                                                cache=cache_history.pop())
      
      dout, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)] = FastConv.backward(dout=dout,
                                                                               cache=cache_history.pop())
      grads['W%d' % (i + 1)] += 2 * reg * self.params['W%d' % (i + 1)]
    
      i -= 1
    
    return loss, grads


def find_overfit_parameters():
  # The following parameters are found by trial and error so the model achieves 100% training accuracy within 30 epochs
  weight_scale = 2e-01
  learning_rate = 8e-04

  return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict: dict,
                                         dtype,
                                         device: str):
  input_dims = data_dict['X_train'].shape[1:]

  data = {
    'X_train': data_dict['X_train'],
    'y_train': data_dict['y_train'],
    'X_val': data_dict['X_val'],
    'y_val': data_dict['y_val'],
  }

  # Hyperparameters
  num_epochs = 50
  lr = 2.0e-3

  # Train the best DeepConvNet that you can on CIFAR-10 within 60 seconds
  model = DeepConvNet(input_dims=input_dims, num_classes=10,
                          num_filters=[32, 32, 64, 64, 128, 128],
                          max_pools=[1, 3, 5],
                          weight_scale='kaiming',
                          batchnorm=False,
                          reg=1.5e-5, dtype=dtype,
                          device=device)
  
  solver = Solver(model=model,
                  data=data,
                  num_epochs=num_epochs,
                  batch_size=128,
                  update_rule=adam,
                  optim_config={'learning_rate': lr},
                  lr_decay = 0.9,
                  verbose=False,
                  device='cuda')
  
  return solver


def kaiming_initializer(Din: int,
                        Dout: int,
                        K: int=None,
                        relu: bool=True,
                        device: str='cpu',
                        dtype=torch.float32):
  """
    Implement Kaiming initialization for linear and convolution layers.
    
    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions for
      this layer
    - K: If K is None, then initialize weights for a linear layer with Din input
      dimensions and Dout output dimensions. Otherwise if K is a nonnegative
      integer then initialize the weights for a convolution layer with Din input
      channels, Dout output channels, and a kernel size of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
      a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
      with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer. For a
      linear layer it should have shape (Din, Dout); for a convolution layer it
      should have shape (Dout, Din, K, K).
  """

  if relu:
    gain = 2.0
  else:
    gain = 1.0
  
  weight = None
  
  if K is None:

    # The weight scale is sqrt(gain / fan_in), where gain is 2 if ReLU is followed by the layer
    if relu:
      weight= (2/Din)**(1/2) * torch.randn(Din, Dout, dtype=dtype, device=device)
    else:
      weight= (1/Din)**(1/2) * torch.randn(Din, Dout, dtype=dtype, device=device)

  else:
    
    # The weight scale is sqrt(gain / fan_in), where gain is 2 if ReLU is followed by the layer
    if relu:
      weight= (2/(Din*K*K))**(1/2) * torch.randn(Din, Dout, K, K, dtype=dtype, device=device)
    else:
      weight= (1/(Din*K*K))**(1/2) * torch.randn(Din, Dout, K, K, dtype=dtype, device=device)
  
  return weight 


class BatchNorm(object):

  @staticmethod
  def forward(x: torch.Tensor,
              gamma: float,
              beta: float,
              bn_param: dict):
    """
      Forward pass for batch normalization.

      During training the sample mean and (uncorrected) sample variance are
      computed from minibatch statistics and used to normalize the incoming data.
      During training we also keep an exponentially decaying running mean of the
      mean and variance of each feature, and these averages are used to normalize
      data at test-time.

      At each timestep we update the running averages for mean and variance using
      an exponential decay based on the momentum parameter:

      running_mean = momentum * running_mean + (1 - momentum) * sample_mean
      running_var = momentum * running_var + (1 - momentum) * sample_var

      Note that the batch normalization paper suggests a different test-time
      behavior: they compute sample mean and variance for each feature using a
      large number of training images rather than using a running average. For
      this implementation we have chosen to use running averages instead since
      they do not require an additional estimation step; the PyTorch
      implementation of batch normalization also uses running averages.

      Input:
      - x: Data of shape (N, D)
      - gamma: Scale parameter of shape (D,)
      - beta: Shift paremeter of shape (D,)
      - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features

      Returns a tuple of:
      - out: of shape (N, D)
      - cache: A tuple of values needed in the backward pass
    """
    
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

    # The train-time forward pass for batch normalization
    if mode == 'train':

      # Mean calculation
      mu = torch.mean(x, 0)
      # Variance calculation
      var = 1. / N * torch.sum((x - mu) ** 2, 0)

      # Normalized Data
      x_hat = (x - mu) / ((var + eps)**(1/2))

      # Scale and Shift
      y = gamma * x_hat + beta
      out = y

      # Make the record of means and variances in running parameters
      running_mean = momentum * running_mean + (1 - momentum) * mu
      running_var = momentum * running_var + (1 - momentum) * var

      # Cache
      cache = (x_hat, mu, var, eps, gamma, beta, x)
    
    # The test-time forward pass for batch normalization
    elif mode == 'test':

      # Normalized Data
      x_hat = (x - running_mean) / ((running_var + eps)**(1/2))
      
      # Scale and Shift
      y = gamma * x_hat + beta
      out = y

      cache = None
    
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean.detach()
    bn_param['running_var'] = running_var.detach()

    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    """
      Backward pass for batch normalization.

      For this implementation, you should write out a computation graph for
      batch normalization on paper and propagate gradients backward through
      intermediate nodes.

      Inputs:
      - dout: Upstream derivatives, of shape (N, D)
      - cache: Variable of intermediates from batchnorm_forward.

      Returns a tuple of:
      - dx: Gradient with respect to inputs x, of shape (N, D)
      - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
      - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """

    # Extract cache
    x_hat, mu, var, eps, gamma, beta, x = cache

    # Shapes
    N, D = dout.shape

    dbeta = torch.sum(dout, 0)
    dgamma = torch.sum(dout * x_hat, 0)

    # The backward pass for batch normalization
    dx_hat = dout * gamma
    dxmu1 = dx_hat * 1 / (var + eps)**(1/2)

    divar = torch.sum(dx_hat * (x - mu), 0)
    dvar = divar * -1 / 2 * (var + eps) ** (-3/2)
    
    dsq = 1 / N * torch.ones((N, D)).to(x.dtype).to(x.device) * dvar
    
    dxmu2 = 2 * (x - mu) * dsq
    
    dx1 = dxmu1 + dxmu2

    dmu = -1 * torch.sum(dxmu1 + dxmu2, 0)
    dx2 = 1 / N * torch.ones((N, D)).to(x.dtype).to(x.device)* dmu

    dx = dx1 + dx2

    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout: torch.Tensor,
                   cache: tuple):
    """
      Alternative backward pass for batch normalization.
      For this implementation you should work out the derivatives for the batch
      normalizaton backward pass on paper and simplify as much as possible. You
      should be able to derive a simple expression for the backward pass. 
      See the jupyter notebook for more hints.
      
      Note: This implementation should expect to receive the same cache variable
      as batchnorm_backward, but might not use all of the values in the cache.

      Inputs / outputs: Same as batchnorm_backward
    """

    # Extract hyperparameters are the batch size
    (x_hat, mu, var, eps, gamma, beta, x), N = cache , dout.shape[0]
    
    # The backward pass for batch normalization
    dbeta = dout.sum(0)
    dgamma = (dout * x_hat).sum(0)
    dx = (dout * gamma - (dout * gamma).sum(0) / N - (dout * gamma * x_hat).sum(0) * x_hat / N) / var**(1 / 2)

    return dx, dgamma, dbeta


class SpatialBatchNorm(object):

  @staticmethod
  def forward(x: torch.Tensor,
              gamma: float,
              beta: float,
              bn_param: dict):
    """
      Computes the forward pass for spatial batch normalization.

      Inputs:
      - x: Input data of shape (N, C, H, W)
      - gamma: Scale parameter, of shape (C,)
      - beta: Shift parameter, of shape (C,)
      - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
        - running_mean: Array of shape (C,) giving running mean of features
        - running_var Array of shape (C,) giving running variance of features

      Returns a tuple of:
      - out: Output data, of shape (N, C, H, W)
      - cache: Values needed for the backward pass
    """

    # Sizes: batch, channel, height, weight
    N, C, H, W = x.shape
    
    # Reshape
    x = x.permute(0, 2, 3, 1).reshape(N * H * W, C)

    # The forward pass for spatial batch normalization
    out, cache = BatchNorm.forward(x=x,
                                   gamma=gamma,
                                   beta=beta,
                                   bn_param=bn_param)

    # Reshape to original input format
    out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)

    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    """
      Computes the backward pass for spatial batch normalization.
      Inputs:
      - dout: Upstream derivatives, of shape (N, C, H, W)
      - cache: Values from the forward pass
      Returns a tuple of:
      - dx: Gradient with respect to inputs, of shape (N, C, H, W)
      - dgamma: Gradient with respect to scale parameter, of shape (C,)
      - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """

    # Sizes: batch, channel, height, weight
    N, C, H, W = dout.shape

    # Reshape
    dout = dout.permute(0, 2, 3, 1).reshape(N * H * W, C)

    # The backward pass for spatial batch normalization
    dx, dgamma, dbeta = BatchNorm.backward_alt(dout=dout,
                                               cache=cache)
    
    # Reshape to original input format
    dx = dx.reshape(N, H, W, C).permute(0, 3, 1, 2)

    return dx, dgamma, dbeta


class FastConv(object):

  @staticmethod
  def forward(x: torch.Tensor,
              w: torch.Tensor,
              b: torch.Tensor,
              conv_param: dict):
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    stride, pad = conv_param['stride'], conv_param['pad']

    layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
    
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)
    
    tx = x.detach().contiguous()
    tx.requires_grad = True

    out = layer(tx)
    cache = (x, w, b, conv_param, tx, out, layer)

    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    
    try:
      x, _, _, _, tx, out, layer = cache

      out.backward(dout)

      dx = tx.grad.detach()
      dw = layer.weight.grad.detach()
      db = layer.bias.grad.detach()

      layer.weight.grad = layer.bias.grad = None

    except RuntimeError:
      dx, dw, db = torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
    
    return dx, dw, db


class FastMaxPool(object):

  @staticmethod
  def forward(x: torch.Tensor,
              pool_param: dict):
    
    N, C, H, W = x.shape
    
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    
    layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
    
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    
    cache = (x, pool_param, tx, out, layer)
    
    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    
    try:
      x, _, tx, out, layer = cache
    
      out.backward(dout)
      dx = tx.grad.detach()
    
    except RuntimeError:
      dx = torch.zeros_like(tx)
    
    return dx


class Conv_ReLU(object):

  @staticmethod
  def forward(x: torch.Tensor,
              w: torch.Tensor,
              b: torch.Tensor,
              conv_param: dict):
    """
      A convenience layer that performs a convolution followed by a ReLU.
      Inputs:
      - x: Input to the convolutional layer
      - w, b, conv_param: Weights and parameters for the convolutional layer
      Returns a tuple of:
      - out: Output from the ReLU
      - cache: Object to give to the backward pass
    """

    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    out, relu_cache = ReLU.forward(a)

    cache = (conv_cache, relu_cache)
    
    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: dict):
    """
      Backward pass for the conv-relu convenience layer.
    """

    conv_cache, relu_cache = cache

    da = ReLU.backward(dout, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)

    return dx, dw, db


class Conv_ReLU_Pool(object):

  @staticmethod
  def forward(x: torch.Tensor,
              w: torch.Tensor,
              b: torch.Tensor,
              conv_param: dict,
              pool_param: dict):
    """
      A convenience layer that performs a convolution, a ReLU, and a pool.
      Inputs:
      - x: Input to the convolutional layer
      - w, b, conv_param: Weights and parameters for the convolutional layer
      - pool_param: Parameters for the pooling layer
      Returns a tuple of:
      - out: Output from the pooling layer
      - cache: Object to give to the backward pass
    """

    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    
    s, relu_cache = ReLU.forward(a)
    
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    
    cache = (conv_cache, relu_cache, pool_cache)
    
    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    """
      Backward pass for the conv-relu-pool convenience layer
    """

    conv_cache, relu_cache, pool_cache = cache

    ds = FastMaxPool.backward(dout, pool_cache)

    da = ReLU.backward(ds, relu_cache)

    dx, dw, db = FastConv.backward(da, conv_cache)

    return dx, dw, db


class Linear_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x: torch.Tensor,
              w: torch.Tensor,
              b: torch.Tensor,
              gamma: float,
              beta: float,
              bn_param: dict):
    """
      Convenience layer that performs an linear transform, batch normalization,
      and ReLU.
      Inputs:
      - x: Array of shape (N, D1); input to the linear layer
      - w, b: Arrays of shape (D2, D2) and (D2,) giving the weight and bias for
        the linear transform.
      - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
        parameters for batch normalization.
      - bn_param: Dictionary of parameters for batch normalization.
      Returns:
      - out: Output from ReLU, of shape (N, D2)
      - cache: Object to give to the backward pass.
    """

    a, fc_cache = Linear.forward(x, w, b)

    a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)

    out, relu_cache = ReLU.forward(a_bn)

    cache = (fc_cache, bn_cache, relu_cache)

    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    """
      Backward pass for the linear-batchnorm-relu convenience layer.
    """

    fc_cache, bn_cache, relu_cache = cache

    da_bn = ReLU.backward(dout, relu_cache)

    da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)

    dx, dw, db = Linear.backward(da, fc_cache)

    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x: torch.Tensor,
              w: torch.Tensor,
              b: torch.Tensor,
              gamma: float,
              beta: float,
              conv_param: dict,
              bn_param: dict):
    
    a, conv_cache = FastConv.forward(x, w, b, conv_param)

    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)

    out, relu_cache = ReLU.forward(an)

    cache = (conv_cache, bn_cache, relu_cache)

    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    
    conv_cache, bn_cache, relu_cache = cache

    dan = ReLU.backward(dout, relu_cache)

    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)

    dx, dw, db = FastConv.backward(da, conv_cache)

    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x: torch.Tensor,
              w: torch.Tensor,
              b: torch.Tensor,
              gamma: float,
              beta: float,
              conv_param: dict,
              bn_param: dict,
              pool_param: dict):
    
    a, conv_cache = FastConv.forward(x, w, b, conv_param)

    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)

    s, relu_cache = ReLU.forward(an)

    out, pool_cache = FastMaxPool.forward(s, pool_param)

    cache = (conv_cache, bn_cache, relu_cache, pool_cache)

    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    
    conv_cache, bn_cache, relu_cache, pool_cache = cache

    ds = FastMaxPool.backward(dout, pool_cache)

    dan = ReLU.backward(ds, relu_cache)

    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)

    dx, dw, db = FastConv.backward(da, conv_cache)
    
    return dx, dw, db, dgamma, dbeta
