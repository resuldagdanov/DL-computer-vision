"""
  Implements a two-layer Neural Network classifier in PyTorch.
  WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
from linear_classifier import sample_batch


def hello_two_layer_net():
  """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
  """
  print('Hello from two_layer_net.py!')


class TwoLayerNet(object):
  def __init__(self,
               input_size: int,
               hidden_size: int,
               output_size: int,
               dtype=torch.float32,
               device: str='cuda',
               std: float=1e-4):
    """
      Initialize the model. Weights are initialized to small random values and
      biases are initialized to zero. Weights and biases are stored in the
      variable self.params, which is a dictionary with the following keys:

      W1: First layer weights; has shape (D, H)
      b1: First layer biases; has shape (H,)
      W2: Second layer weights; has shape (H, C)
      b2: Second layer biases; has shape (C,)

      Inputs:
      - input_size: The dimension D of the input data.
      - hidden_size: The number of neurons H in the hidden layer.
      - output_size: The number of classes C.
      - dtype: Optional, data type of each initial weight params
      - device: Optional, whether the weight params is on GPU or CPU
      - std: Optional, initial weight scaler.
    """

    # reset seed before start
    random.seed(0)
    torch.manual_seed(0)

    self.params = {}
    self.params['W1'] = std * torch.randn(input_size, hidden_size, dtype=dtype, device=device)
    self.params['b1'] = torch.zeros(hidden_size, dtype=dtype, device=device)
    self.params['W2'] = std * torch.randn(hidden_size, output_size, dtype=dtype, device=device)
    self.params['b2'] = torch.zeros(output_size, dtype=dtype, device=device)

  def loss(self,
           X: torch.Tensor,
           y: torch.Tensor=None,
           reg: float=0.0):
    return nn_forward_backward(self.params, X, y, reg)

  def train(self,
            X: torch.Tensor,
            y: torch.Tensor,
            X_val: torch.Tensor,
            y_val: torch.Tensor,
            learning_rate: float=1e-3,
            learning_rate_decay: float=0.95,
            reg: float=5e-6,
            num_iters: int=100,
            batch_size: int=200,
            verbose: bool=False):
    return nn_train(
            self.params,
            nn_forward_backward,
            nn_predict,
            X, y, X_val, y_val,
            learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose)

  def predict(self,
              X: torch.Tensor):
    return nn_predict(self.params, nn_forward_backward, X)

  def save(self,
           path: str):
    torch.save(self.params, path)
    print("Saved in {}".format(path))

  def load(self,
           path: str):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint
    print("load checkpoint file: {}".format(path))


def nn_forward_pass(params: dict,
                    X: torch.Tensor):
    """
      The first stage of our neural network implementation: Run the forward pass
      of the network to compute the hidden layer features and classification
      scores. The network architecture should be:

      FC layer -> ReLU (hidden) -> FC layer (scores)

      As a practice, we will NOT allow to use torch.relu and torch.nn ops
      just for this time (you can use it from A3).

      Inputs:
      - params: a dictionary of PyTorch Tensor that store the weights of a model.
        It should have following keys with shape
            W1: First layer weights; has shape (D, H)
            b1: First layer biases; has shape (H,)
            W2: Second layer weights; has shape (H, C)
            b2: Second layer biases; has shape (C,)
      - X: Input data of shape (N, D). Each X[i] is a training sample.

      Returns a tuple of:
      - scores: Tensor of shape (N, C) giving the classification scores for X
      - hidden: Tensor of shape (N, H) giving the hidden layer representation
        for each input value (after the ReLU).
    """

    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    N, D = X.shape

    # Fully-connected layer with bias
    hidden = torch.matmul(X, W1) + b1

    # Rectified linear unit
    hidden = torch.max(torch.zeros_like(hidden), hidden)

    # Fully-connected layer with bias
    scores = torch.matmul(hidden, W2) + b2

    return scores, hidden


def nn_forward_backward(params: dict,
                        X: torch.Tensor,
                        y: torch.Tensor=None,
                        reg: float=0.0):
    """
      Compute the loss and gradients for a two layer fully connected neural
      network. When you implement loss and gradient, please don't forget to
      scale the losses/gradients by the batch size.

      Inputs: First two parameters (params, X) are same as nn_forward_pass
      - params: a dictionary of PyTorch Tensor that store the weights of a model.
        It should have following keys with shape
            W1: First layer weights; has shape (D, H)
            b1: First layer biases; has shape (H,)
            W2: Second layer weights; has shape (H, C)
            b2: Second layer biases; has shape (C,)
      - X: Input data of shape (N, D). Each X[i] is a training sample.
      - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
        an integer in the range 0 <= y[i] < C. This parameter is optional; if it
        is not passed then we only return scores, and if it is passed then we
        instead return the loss and gradients.
      - reg: Regularization strength.

      Returns:
      If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
      the score for class c on input X[i].

      If y is not None, instead return a tuple of:
      - loss: Loss (data loss and regularization loss) for this batch of training
        samples.
      - grads: Dictionary mapping parameter names to gradients of those parameters
        with respect to the loss function; has the same keys as self.params.
    """

    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    N, D = X.shape

    # Compute the forward pass
    scores, hidden = nn_forward_pass(params, X)

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Calculate the Softmax classifier loss
    exp_scores = torch.exp(scores - torch.max(scores,
                                              dim=1,
                                              keepdim=True).values)
    softmax_scores = exp_scores / torch.sum(exp_scores,
                                            dim=1,
                                            keepdim=True)
    data_loss = -torch.mean(torch.log(softmax_scores[range(N), y]))

    # Add L2 regularization for W1 and W2 to the loss
    reg_loss = reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))
    loss = data_loss + reg_loss

    # Gradient and take average of last output layer output by the batch size
    diff_scores = softmax_scores.clone()
    diff_scores[torch.arange(N), y] -= 1
    diff_scores /= N

    # gradients for second layer weights
    diff_W2 = hidden.t() @ diff_scores
    # gradients for second layer biases
    diff_b2 = torch.sum(diff_scores, dim=0)

    # gradient of ReLU
    diff_hidden = diff_scores @ W2.t()
    diff_hidden[hidden <= 0] = 0

    # gradients for first layer weights
    diff_W1 = X.t() @ diff_hidden
    # gradients for first layer biases
    diff_b1 = torch.sum(diff_hidden, dim=0)

    # Add regularization gradient contribution
    # DO NOT multiply the regularization term by 1/2 (no coefficient)
    diff_W2 += reg * W2
    diff_W1 += reg * W1

    # Store the derivatives of the weights and biases in the dictionary
    grads = {"W1": diff_W1, "b1": diff_b1,
             "W2": diff_W2, "b2": diff_b2}

    return loss, grads


def nn_train(params: dict,
             loss_func,
             pred_func,
             X: torch.Tensor,
             y: torch.Tensor,
             X_val: torch.Tensor,
             y_val: torch.Tensor,
             learning_rate: float=1e-3,
             learning_rate_decay: float=0.95,
             reg: float=5e-6,
             num_iters: int=100,
             batch_size: int=200,
             verbose: bool=False):
  """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients.
      It takes as input:
      - params: Same as input to nn_train
      - X_batch: A minibatch of inputs of shape (B, D)
      - y_batch: Ground-truth labels for X_batch
      - reg: Same as input to nn_train
      And it returns a tuple of:
        - loss: Scalar giving the loss on the minibatch
        - grads: Dictionary mapping parameter names to gradients of the loss with
          respect to the corresponding parameter.
    - pred_func: prediction function that im
    - X: A PyTorch tensor of shape (N, D) giving training data.
    - y: A PyTorch tensor f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
    - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Returns: A dictionary giving statistics about the training process
  """

  num_train = X.shape[0]
  iterations_per_epoch = max(num_train // batch_size, 1)

  # Use SGD to optimize the parameters in self.model
  loss_history = []
  train_acc_history = []
  val_acc_history = []

  for it in range(num_iters):
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

    # Compute loss and gradients using the current minibatch
    loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
    loss_history.append(loss.item())

    # Update the parameters of the network
    for param_name, grad in grads.items():
        params[param_name] -= learning_rate * grad
    
    if verbose and it % 100 == 0:
      print('iteration %d / %d: loss %f' % (it, num_iters, loss.item()))

    # Every epoch, check train and val accuracy and decay learning rate.
    if it % iterations_per_epoch == 0:
      # Check accuracy
      y_train_pred = pred_func(params, loss_func, X_batch)
      train_acc = (y_train_pred == y_batch).float().mean().item()
      y_val_pred = pred_func(params, loss_func, X_val)
      val_acc = (y_val_pred == y_val).float().mean().item()
      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)

      # Decay learning rate
      learning_rate *= learning_rate_decay

  return {
    'loss_history': loss_history,
    'train_acc_history': train_acc_history,
    'val_acc_history': val_acc_history,
  }


def nn_predict(params: dict,
               loss_func,
               X: torch.Tensor):
  """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients
    - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
  """

  # Compute the forward pass
  scores, hidden = nn_forward_pass(params, X)

 # Prediction
  y_pred = torch.argmax(scores, dim=1)

  return y_pred


def nn_get_search_params():
  """
    Return candidate hyperparameters for a TwoLayerNet model.
    You should provide at least two param for each, and total grid search
    combinations should be less than 256. If not, it will take
    too much time to train on such hyperparameter combinations.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - hidden_sizes: hidden value sizes, e.g. [8, 16, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    - learning_rate_decays: learning rate decay candidates
                                e.g. [1.0, 0.95, ...]
  """

  # Add your own hyper parameter lists
  learning_rates = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e0]
  regularization_strengths = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-5]
  hidden_sizes = [16, 32, 64, 128, 256, 512, 1024]
  learning_rate_decays = [1.0, 0.99, 0.98, 0.97, 0.95, 0.92]

  return learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays


def find_best_net(data_dict: dict,
                  get_param_set_fn):
  """
    Tune hyperparameters using the validation set.
    Store your best trained TwoLayerNet model in best_net, with the return value
    of ".train()" operation in best_stat and the validation accuracy of the
    trained best model in best_val_acc. Your hyperparameters should be received
    from in nn_get_search_params

    Inputs:
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - get_param_set_fn (function): A function that provides the hyperparameters
                                  (e.g., nn_get_search_params)
                                  that gives (learning_rates, hidden_sizes,
                                  regularization_strengths, learning_rate_decays)
                                  You should get hyperparameters from
                                  get_param_set_fn.

    Returns:
    - best_net (instance): a trained TwoLayerNet instances with
                          (['X_train', 'y_train'], batch_size, learning_rate,
                          learning_rate_decay, reg)
                          for num_iter times.
    - best_stat (dict): return value of "best_net.train()" operation
    - best_val_acc (float): validation accuracy of the best_net
  """

  num_classes = 10
  num_iters = 3000

  best_net = None
  best_stat = None
  best_val_acc = 0.0

  # Extract data dictionary
  X_train, y_train = data_dict["X_train"], data_dict["y_train"]
  X_val, y_val = data_dict["X_val"], data_dict["y_val"]

  # Get possible hyperparameters to check whether the best or not
  learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays = get_param_set_fn()

  # Iterate over possible candidate hyperparameters
  for learnings in learning_rates:
      for hiddens in hidden_sizes:
          for regularizations in regularization_strengths:
              for decays in learning_rate_decays:

                # Create two layer neural network
                network = TwoLayerNet(input_size=X_train.shape[1],
                                      hidden_size=hiddens,
                                      output_size=num_classes,
                                      dtype=X_train.dtype,
                                      device=X_train.device)
              
                # Train the network
                statistics = network.train(X=X_train,
                                           y=y_train,
                                           X_val=X_val,
                                           y_val=y_val,
                                           learning_rate=learnings,
                                           learning_rate_decay=decays,
                                           reg=regularizations,
                                           num_iters=num_iters)
                
                # Evaluate the trained neural network model
                y_val_pred = network.predict(X_val)
                val_accuracy = 100 * (y_val_pred == y_val).double().mean().item()
                
                # Check if current hyperparameters give better accuracy
                if val_accuracy > best_val_acc:
                    best_net = network
                    best_stat = statistics
                    best_val_acc = val_accuracy
  
  return best_net, best_stat, best_val_acc
