"""
  Implements fully connected networks in PyTorch.
  WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
from a3_helper import svm_loss, softmax_loss
from eecs598 import Solver


def hello_fully_connected_networks():
  """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
  """
  print('Hello from fully_connected_networks.py!')


class Linear(object):

  @staticmethod
  def forward(x: torch.Tensor,
              w: torch.Tensor,
              b: torch.Tensor):
    """
      Computes the forward pass for an linear (fully-connected) layer.
      The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
      examples, where each example x[i] has shape (d_1, ..., d_k). We will
      reshape each input into a vector of dimension D = d_1 * ... * d_k, and
      then transform it to an output vector of dimension M.
      Inputs:
      - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
      - w: A tensor of weights, of shape (D, M)
      - b: A tensor of biases, of shape (M,)
      Returns a tuple of:
      - out: output, of shape (N, M)
      - cache: (x, w, b)
    """
    # reshape to (N, D)
    out=torch.reshape(x,(x.shape[0], -1)).mm(w) + b

    cache = (x, w, b)
    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: tuple):
    """
      Computes the backward pass for an linear layer.
      Inputs:
      - dout: Upstream derivative, of shape (N, M)
      - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)
        - b: Biases, of shape (M,)
      Returns a tuple of:
      - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
      - dw: Gradient with respect to w, of shape (D, M)
      - db: Gradient with respect to b, of shape (M,)
    """

    x, w, b = cache
    
    dx = torch.reshape(dout.mm(w.t()),x.shape)
    dw = torch.reshape(x,(x.shape[0], -1)).t().mm(dout)
    db = torch.sum(dout, 0)

    return dx, dw, db


class ReLU(object):

  @staticmethod
  def forward(x: torch.Tensor):
    """
      Computes the forward pass for a layer of rectified linear units (ReLUs).
      Input:
      - x: Input; a tensor of any shape
      Returns a tuple of:
      - out: Output, a tensor of the same shape as x
      - cache: x
    """

    # apply non-linear max operation
    out = torch.max(torch.zeros_like(x), x)
   
    cache = x
    
    return out, cache

  @staticmethod
  def backward(dout: torch.Tensor,
               cache: torch.Tensor):
    """
      Computes the backward pass for a layer of rectified linear units (ReLUs).
      Input:
      - dout: Upstream derivatives, of any shape
      - cache: Input x, of same shape as dout
      Returns:
      - dx: Gradient with respect to x
    """

    # apply the derivative of the ReLU function
    dx = torch.where(cache > 0, dout, torch.zeros_like(dout))
    
    return dx


class Linear_ReLU(object):

  @staticmethod
  def forward(x: torch.Tensor,
              w: torch.Tensor,
              b: torch.Tensor):
    """
      Convenience layer that performs an linear transform followed by a ReLU.

      Inputs:
      - x: Input to the linear layer
      - w, b: Weights for the linear layer
      Returns a tuple of:
      - out: Output from the ReLU
      - cache: Object to give to the backward pass
    """

    a, fc_cache = Linear.forward(x, w, b)
    out, relu_cache = ReLU.forward(a)

    cache = (fc_cache, relu_cache)

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
      Backward pass for the linear-relu convenience layer
    """

    fc_cache, relu_cache = cache
    da = ReLU.backward(dout, relu_cache)

    dx, dw, db = Linear.backward(da, fc_cache)

    return dx, dw, db


class TwoLayerNet(object):
  """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
  """

  def __init__(self,
               input_dim: int=3*32*32,
               hidden_dim: int=100,
               num_classes: int=10,
               weight_scale: float=1e-3,
               reg: float=0.0,
               dtype=torch.float32,
               device: str='cpu'):
    """
      Initialize a new network.
      Inputs:
      - input_dim: An integer giving the size of the input
      - hidden_dim: An integer giving the size of the hidden layer
      - num_classes: An integer giving the number of classes to classify
      - weight_scale: Scalar giving the standard deviation for random
        initialization of the weights.
      - reg: Scalar giving L2 regularization strength.
      - dtype: A torch data type object; all computations will be performed using
        this datatype. float is faster but less accurate, so you should use
        double for numeric gradient checking.
      - device: device to use for computation. 'cpu' or 'cuda'
    """

    self.params = {}
    self.reg = reg

    # initialize weights and biases for the first layer
    self.params['W1'] = weight_scale * torch.randn(input_dim, hidden_dim, dtype=dtype, device=device)
    self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)

    # initialize weights and biases for the second layer
    self.params['W2'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
    self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)

  def save(self,
           path: str):
    checkpoint = {
      'reg': self.reg,
      'params': self.params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self,
           path: str,
           dtype,
           device: str):
    checkpoint = torch.load(path, map_location='cpu')

    self.params = checkpoint['params']
    self.reg = checkpoint['reg']
    
    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)
    print("load checkpoint file: {}".format(path))

  def loss(self,
           X: torch.Tensor,
           y: torch.Tensor=None):
    """
      Compute loss and gradient for a minibatch of data.

      Inputs:
      - X: Tensor of input data of shape (N, d_1, ..., d_k)
      - y: int64 Tensor of labels, of shape (N,). y[i] gives the label for X[i].

      Returns:
      If y is None, then run a test-time forward pass of the model and return:
      - scores: Tensor of shape (N, C) giving classification scores, where
        scores[i, c] is the classification score for X[i] and class c.
      If y is not None, then run a training-time forward and backward pass and
      return a tuple of:
      - loss: Scalar value giving the loss
      - grads: Dictionary with the same keys as self.params, mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    scores = None

    # unpack variables
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    # forward pass
    # shape: (N, H)
    h1 = X.view(X.shape[0], -1).mm(W1) + b1
    # shape: (N, H)
    h1_relu = torch.maximum(h1, torch.zeros_like(h1))
    # shape: (N, C)
    scores = h1_relu.mm(W2) + b2

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}

    # compute loss and d_scores
    exp_scores = torch.exp(scores)
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

    num_examples = X.shape[0]

    # add regularization to the loss output
    data_loss = torch.mean(-torch.log(probs[torch.arange(num_examples), y]))
    reg_loss = self.reg * (torch.sum(W1 ** 2) + torch.sum(W2 ** 2))

    # add neural network ouput loss value with regularization value
    loss = data_loss + reg_loss

    # compute gradients for backward pass
    dscores = probs
    dscores[torch.arange(num_examples), y] -= 1
    dscores /= num_examples

    # store gradients in the grads dictionary
    grads['W2'] = h1_relu.T.mm(dscores) + 2 * self.reg * W2
    grads['b2'] = torch.sum(dscores, dim=0)

    dhidden = dscores.mm(W2.T)
    dhidden[h1 <= 0] = 0

    grads['W1'] = X.view(num_examples, -1).T.mm(dhidden) + 2 * self.reg * W1
    grads['b1'] = torch.sum(dhidden, dim=0)

    return loss, grads


class FullyConnectedNet(object):
  """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self,
               hidden_dims: list,
               input_dim: int=3*32*32,
               num_classes: int=10,
               dropout: float=0.0,
               reg: float=0.0,
               weight_scale: float=1e-2,
               seed: int=None,
               dtype=torch.float,
               device: str='cpu'):
    """
      Initialize a new FullyConnectedNet.

      Inputs:
      - hidden_dims: A list of integers giving the size of each hidden layer.
      - input_dim: An integer giving the size of the input.
      - num_classes: An integer giving the number of classes to classify.
      - dropout: Scalar between 0 and 1 giving the drop probability for networks
        with dropout. If dropout=0 then the network should not use dropout.
      - reg: Scalar giving L2 regularization strength.
      - weight_scale: Scalar giving the standard deviation for random
        initialization of the weights.
      - seed: If not None, then pass this random seed to the dropout layers. This
        will make the dropout layers deteriminstic so we can gradient check the
        model.
      - dtype: A torch data type object; all computations will be performed using
        this datatype. float is faster but less accurate, so you should use
        double for numeric gradient checking.
      - device: device to use for computation. 'cpu' or 'cuda'
    """

    self.use_dropout = dropout != 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # initialize weights and biases for the first layer
    self.params['W1'] = torch.randn(input_dim, hidden_dims[0], dtype=dtype, device=device) * weight_scale
    self.params['b1'] = torch.zeros(hidden_dims[0], dtype=dtype, device=device)

    # initialize weights and biases for the hidden layers
    for i in range(1, len(hidden_dims)):
      self.params[f'W{i+1}'] = torch.randn(hidden_dims[i-1], hidden_dims[i], dtype=dtype, device=device) * weight_scale
      self.params[f'b{i+1}'] = torch.zeros(hidden_dims[i], dtype=dtype, device=device)
    
    # initialize weights and biases for the output layer
    self.params[f'W{self.num_layers}'] = torch.randn(hidden_dims[-1], num_classes, dtype=dtype, device=device) * weight_scale
    self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, dtype=dtype, device=device)

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      
      if seed is not None:
        self.dropout_param['seed'] = seed

  def save(self,
           path: str):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'use_dropout': self.use_dropout,
      'dropout_param': self.dropout_param,
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
    self.use_dropout = checkpoint['use_dropout']
    self.dropout_param = checkpoint['dropout_param']

    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))

  def loss(self,
           X: torch.Tensor,
           y: torch.Tensor=None):
    """
      Compute loss and gradient for the fully-connected net.
      Input / output: Same as TwoLayerNet above.
    """

    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.use_dropout:
      self.dropout_param['mode'] = mode
    
    scores = None
    out = X
    caches = []

    # forward pass for the fully-connected net
    for i in range(self.num_layers - 1):
      W = self.params['W' + str(i + 1)]
      b = self.params['b' + str(i + 1)]

      out, cache = Linear_ReLU.forward(x=out,
                                       w=W,
                                       b=b)
      caches.append(cache)
      
      # when using a dropout layer
      if self.use_dropout:
        out, cache = Dropout.forward(x=out,
                                     dropout_param=self.dropout_param)
        caches.append(cache)
    
    W = self.params['W' + str(self.num_layers)]
    b = self.params['b' + str(self.num_layers)]

    scores, cache = Linear.forward(x=out,
                                   w=W,
                                   b=b)
    caches.append(cache)

    # if test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    # compute data loss using softmax
    loss, dscores = softmax_loss(x=scores,
                                 y=y)
    
    # backward pass for the fully-connected net
    dout, dw, db = Linear.backward(dout=dscores,
                                   cache=caches.pop())

    grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
    grads['b' + str(self.num_layers)] = db

    for i in range(self.num_layers-1, 0, -1):
        
        if self.use_dropout:
            dout = Dropout.backward(dout=dout,
                                    cache=caches.pop())
        
        dout, dw, db = Linear_ReLU.backward(dout=dout,
                                            cache=caches.pop())

        grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
        grads['b' + str(i)] = db

    loss += 0.5 * self.reg * sum([torch.sum(self.params['W' + str(i)] ** 2)
                                  for i in range(1, self.num_layers+1)])

    return loss, grads


def create_solver_instance(data_dict: dict,
                           dtype,
                           device: str):
  # create a model object
  model = TwoLayerNet(hidden_dim=1024,
                      weight_scale=1e-2,
                      reg=1e-3,
                      dtype=dtype,
                      device=device)
  
  # create Solver instance
  solver = Solver(model=model,
                  data=data_dict,
                  num_epochs=50,
                  device=device)
  
  return solver


def get_three_layer_network_params():
  # change weight_scale and learning_rate so your model achieves 100%
  # training accuracy within 20 epochs
  weight_scale = 1e-1
  learning_rate = 5e-1
  
  return weight_scale, learning_rate


def get_five_layer_network_params():
  # change weight_scale and learning_rate so your model achieves 100%
  # training accuracy within 20 epochs
  learning_rate = 2e-1
  weight_scale = 1e-1

  return weight_scale, learning_rate


def sgd(w: torch.Tensor,
        dw: torch.Tensor,
        config: dict=None):
    """
      Performs vanilla stochastic gradient descent.
      config format:
      - learning_rate: Scalar learning rate.
    """

    if config is None:
      config = {}
    
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw

    return w, config


def sgd_momentum(w: torch.Tensor,
                 dw: torch.Tensor,
                 config: dict=None):
  """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
  """

  if config is None:
    config = {}
  
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  
  v = config.get('velocity', torch.zeros_like(w))

  next_w = None

  # implement the momentum update formula
  mu = config['momentum']
  lr = config['learning_rate']

  # also use and update the velocity v
  v = mu * v - lr * dw
  next_w = w + v
  
  config['velocity'] = v

  return next_w, config


def rmsprop(w: torch.Tensor,
            dw: torch.Tensor,
            config: dict=None):
  """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
  """

  if config is None:
    config = {}
  
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', torch.zeros_like(w))

  next_w = None

  # get config values
  learning_rate = config['learning_rate']
  decay_rate = config['decay_rate']
  eps = config['epsilon']
  cache = config['cache']

  # compute the squared gradient
  cache = decay_rate * cache + (1 - decay_rate) * torch.pow(dw, 2)

  # implement the RMSprop update formula to update the parameters
  next_w = w - learning_rate * dw / (torch.sqrt(cache) + eps)
  
  # update the cache in the config dictionary
  config['cache'] = cache

  return next_w, config


def adam(w: torch.Tensor,
         dw: torch.Tensor,
         config: dict=None):
  """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
  """

  if config is None:
    config = {}

  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', torch.zeros_like(w))
  config.setdefault('v', torch.zeros_like(w))
  config.setdefault('t', 0)

  next_w = None

  # update iteration number
  config['t'] += 1

  # implement the Adam update formula
  m, v, t = config['m'], config['v'], config['t']
  beta1, beta2, eps, lr = config['beta1'], config['beta2'], config['epsilon'], config['learning_rate']
  
  m = beta1 * m + (1 - beta1) * dw
  v = beta2 * v + (1 - beta2) * (dw ** 2)
  m_hat = m / (1 - beta1 ** t)
  v_hat = v / (1 - beta2 ** t)

  # perform Adam update
  next_w = w - lr * m_hat / (torch.sqrt(v_hat) + eps)

  # store updated moving averages and iteration number in config
  config['m'], config['v'] = m, v

  return next_w, config


class Dropout(object):

  @staticmethod
  def forward(x: torch.Tensor,
              dropout_param: dict):
    """
      Performs the forward pass for (inverted) dropout.
      Inputs:
      - x: Input data: tensor of any shape
      - dropout_param: A dictionary with the following keys:
        - p: Dropout parameter. We *drop* each neuron output with probability p.
        - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
        - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.
      Outputs:
      - out: Tensor of the same shape as x.
      - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
        mask that was used to multiply the input; in test mode, mask is None.
      NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
      See http://cs231n.github.io/neural-networks-2/#reg for more details.
      NOTE 2: Keep in mind that p is the probability of **dropping** a neuron
      output; this might be contrary to some sources, where it is referred to
      as the probability of keeping a neuron output.
    """

    p, mode = dropout_param['p'], dropout_param['mode']

    if 'seed' in dropout_param:
      torch.manual_seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
      # store the dropout mask in the mask variable
      mask = (torch.rand_like(x) < p).float() / p
      out = x * mask
    
    elif mode == 'test':
      out = x

    cache = (dropout_param, mask)

    return out, cache


  @staticmethod
  def backward(dout: torch.Tensor,
               cache: torch.Tensor):
    """
      Perform the backward pass for (inverted) dropout.
      Inputs:
      - dout: Upstream derivatives, of any shape
      - cache: (dropout_param, mask) from Dropout.forward.
    """

    dropout_param, mask = cache

    mode = dropout_param['mode']
    p = dropout_param['p']

    if mode == 'train':
      dx = dout * mask / p
    
    elif mode == 'test':
      dx = dout
    
    else:
      dx = None
    
    return dx
