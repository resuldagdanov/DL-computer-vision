"""
  Implements linear classifeirs in PyTorch.
  WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
from abc import abstractmethod


def hello_linear_classifier():
  """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
  """
  print('Hello from linear_classifier.py!')


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier(object):
  """
    An abstarct class for the linear classifiers
  """
  
  # Note: We will re-use `LinearClassifier' in both SVM and Softmax
  def __init__(self):
    random.seed(0)
    torch.manual_seed(0)
    
    self.W = None

  def train(self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            learning_rate: float=1e-3,
            reg: float=1e-5,
            num_iters: int=100,
            batch_size: int=200,
            verbose=False):
    
    train_args = (self.loss,
                  self.W,
                  X_train,
                  y_train,
                  learning_rate,
                  reg,
                  num_iters,
                  batch_size,
                  verbose)
    
    self.W, loss_history = train_linear_classifier(*train_args)
    
    return loss_history

  def predict(self,
              X: torch.Tensor):
    return predict_linear_classifier(self.W, X)

  @abstractmethod
  def loss(self,
           W: torch.Tensor,
           X_batch: torch.Tensor,
           y_batch: torch.Tensor,
           reg: float):
    """
      Compute the loss function and its derivative.
      Subclasses will override this.

      Inputs:
      - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
      - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
      - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
      - reg: (float) regularization strength.

      Returns: A tuple containing:
      - loss as a single float
      - gradient with respect to self.W; an tensor of the same shape as W
    """

    raise NotImplementedError

  def _loss(self,
            X_batch: torch.Tensor,
            y_batch: torch.Tensor,
            reg: float):
    self.loss(self.W, X_batch, y_batch, reg)

  def save(self,
           path: str):
    torch.save({'W': self.W}, path)
    print("Saved in {}".format(path))

  def load(self,
           path: str):
    W_dict = torch.load(path, map_location='cpu')

    self.W = W_dict['W']
    print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
  """
    A subclass that uses the Multiclass SVM loss function
  """

  def loss(self,
           W: torch.Tensor,
           X_batch: torch.Tensor,
           y_batch: torch.Tensor,
           reg: float):
    return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """
    A subclass that uses the Softmax + Cross-entropy loss function
  """
  
  def loss(self,
           W: torch.Tensor,
           X_batch: torch.Tensor,
           y_batch: torch.Tensor,
           reg: float):
    return softmax_loss_vectorized(W, X_batch, y_batch, reg)


def svm_loss_naive(W: torch.Tensor,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   reg: float):
  """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implment the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
  """

  # initialize the gradient as zero
  dW = torch.zeros_like(W)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  loss = 0.0
  
  for i in range(num_train):
    scores = W.t().mv(X[i])
    correct_class_score = scores[y[i]]
    
    for j in range(num_classes):
      if j == y[i]:
        continue
      
      # note delta = 1
      margin = scores[j] - correct_class_score + 1
      
      if margin > 0:
        loss += margin
      
        # Compute the gradient of the loss function and store it dW. (part 1)
        # Rather than first computing the loss and then computing the
        # derivative, it is simple to compute the derivative at the same time
        # that the loss is being computed.
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * torch.sum(W * W)

  # Compute the gradient of the loss function and store it in dW. (part 2)
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W: torch.Tensor,
                        X: torch.Tensor,
                        y: torch.Tensor,
                        reg: float):
  """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
  """

  # initialize the gradient as zero
  dW = torch.zeros_like(W)

  loss = 0.0

  # Implement a vectorized version of the structured SVM loss, storing the result in loss.
  N, D = X.shape
  C = W.shape[1]

  # shape: (N, C)
  scores = X.mm(W)
  # shape: (N,)
  correct_class_scores = scores[torch.arange(N), y]

  # shape: (N, C)
  margins = scores - correct_class_scores.view(-1, 1) + 1
  # set the margin to 0 for the correct class
  margins[torch.arange(N), y] = 0

  # Compute the loss
  loss = torch.sum(torch.relu(margins))
  loss /= N
  loss += reg * torch.sum(W * W)

  # Compute the gradient
  binary = margins > 0
  binary = binary.float()
  row_sum = torch.sum(binary, dim=1)
  binary[torch.arange(N), y] = -row_sum

  # Implement a vectorized version of the gradient for the structured SVM
  # loss, storing the result in dW.
  dW = X.t().mm(binary.double())
  dW /= N
  dW += reg * 2 * W.double()

  return loss, dW


def sample_batch(X: torch.Tensor,
                 y: torch.Tensor,
                 num_train: int,
                 batch_size: int):
  """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
  """

  # Store the data in X_batch and their corresponding labels in
  # y_batch; after sampling, X_batch should have shape (batch_size, dim)
  # and y_batch should have shape (batch_size,)
  idx = torch.randint(low=0, high=num_train, size=(batch_size,))

  # Use the indices to select a random subset of the data and labels
  X_batch = X[idx]
  y_batch = y[idx]

  return X_batch, y_batch


def train_linear_classifier(loss_func,
                            W: torch.Tensor,
                            X: torch.Tensor,
                            y: torch.Tensor,
                            learning_rate: float=1e-3,
                            reg: float=1e-5,
                            num_iters: int=100,
                            batch_size: int=200,
                            verbose: bool=False):
  """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
  """

  # assume y takes values 0...K-1 where K is number of classes
  num_train, dim = X.shape

  if W is None:
    # lazily initialize W
    num_classes = torch.max(y) + 1
    W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)
  
  else:
    num_classes = W.shape[1]
  
  # Run stochastic gradient descent to optimize W
  loss_history = []
  
  for it in range(num_iters):
    # implement sample_batch function
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

    # evaluate loss and gradient
    loss, grad = loss_func(W, X_batch, y_batch, reg)
    loss_history.append(loss.item())

    # Perform parameter update
    # Update the weights using the gradient and the learning rate.
    W -= learning_rate * grad

    if verbose and it % 100 == 0:
      print('iteration %d / %d: loss %f' % (it, num_iters, loss))

  return W, loss_history


def predict_linear_classifier(W: torch.Tensor,
                              X: torch.Tensor):
  """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
      elemment of X. Each element of y_pred should be between 0 and C - 1.
  """

  # Implement this method. Store the predicted labels in y_pred
  scores = X.mm(W)
  y_pred = torch.argmax(scores, dim=1)
  
  return y_pred


def svm_get_search_params():
  """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
  """

  # add your own hyper parameter lists. 
  learning_rates = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
  regularization_strengths = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

  return learning_rates, regularization_strengths


def test_one_param_set(cls: LinearClassifier,
                       data_dict: dict,
                       lr: float,
                       reg: float,
                       num_iters: int=2000):
  """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
  """

  # The accuracy is simply the fraction of data points
  # that are correctly classified.
  train_acc = 0.0
  val_acc = 0.0

  # Train a linear SVM on the training set
  loss_history = cls.train(X_train=data_dict['X_train'],
                           y_train=data_dict['y_train'],
                           learning_rate=lr,
                           reg=reg,
                           num_iters=num_iters)
  
  # Get pridiction results
  y_train_pred = cls.predict(data_dict['X_train'])
  y_val_pred = cls.predict(data_dict['X_val'])

  # Compute trained accuracy on the training and validation sets
  train_acc = (y_train_pred == data_dict['y_train']).float().mean().item()
  val_acc = (y_val_pred == data_dict['y_val']).float().mean().item()

  return cls, train_acc, val_acc


def softmax_loss_naive(W: torch.Tensor,
                       X: torch.Tensor,
                       y: torch.Tensor,
                       reg: float):
  """
    Softmax loss function, naive implementation (with loops).  When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
  """

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  # Compute the loss and gradient using explicit loops
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i] @ W # X[i].view(-1, 1).mm(W.T)
    scores -= torch.max(scores)  # For numerical stability

    exp_scores = torch.exp(scores)
    probs = exp_scores / torch.sum(exp_scores)

    loss += -torch.log(probs[y[i]])

    # Compute the gradient
    dscores = probs.clone()
    dscores[y[i]] -= 1
    dW += X[i].view(-1, 1) @ dscores.view(1, -1) # X[i].view(-1, 1).mm(dscores.view(1, -1))

  # Average the loss and gradient
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and gradient
  loss += reg * torch.sum(W * W)
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W: torch.Tensor,
                            X: torch.Tensor,
                            y: torch.Tensor,
                            reg: int):
  """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
  """

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  num_train = X.shape[0]

  # Compute the scores and the softmax loss
  scores = X @ W
  scores -= scores.max(dim=1, keepdim=True).values  # For numerical stability
  exp_scores = torch.exp(scores)
  softmax_probs = exp_scores / exp_scores.sum(dim=1, keepdim=True)
  loss = -torch.log(softmax_probs[range(num_train), y]).sum()

  # Compute the gradient
  softmax_probs[range(num_train), y] -= 1
  dW = X.T @ softmax_probs

  # Divide by the number of training examples
  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += reg * (W ** 2).sum()
  dW += 2 * reg * W

  return loss, dW


def softmax_get_search_params():
  """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
  """
  
  # add your own hyper parameter lists. 
  learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  regularization_strengths = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

  return learning_rates, regularization_strengths
