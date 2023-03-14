"""
  Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
import math
import statistics
import numpy as np


def hello():
  """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
  """
  print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor,
                                x_test: torch.Tensor):
  """
    Computes the squared Euclidean distance between each element of the training
    set and each element of the test set. Images should be flattened and treated
    as vectors.

    This implementation uses a naive set of nested loops over the training and
    test data.

    The input data may have any number of dimensions -- for example this function
    should be able to compute nearest neighbor between vectors, in which case
    the inputs will have shape (num_{train, test}, D); it should also be able to
    compute nearest neighbors between images, where the inputs will have shape
    (num_{train, test}, C, H, W). More generally, the inputs will have shape
    (num_{train, test}, D1, D2, ..., Dn); you should flatten each element
    of shape (D1, D2, ..., Dn) into a vector of shape (D1 * D2 * ... * Dn) before
    computing distances.

    The input tensors should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
    You may not use any functions from torch.nn or torch.nn.functional.

    Inputs:
    - x_train: Torch tensor of shape (num_train, D1, D2, ...)
    - x_test: Torch tensor of shape (num_test, D1, D2, ...)

    Returns:
    - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
      squared Euclidean distance between the ith training point and the jth test
      point. It should have the same dtype as x_train.
  """

  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]

  dists = x_train.new_zeros(num_train, num_test)

  flattened_x_train = torch.flatten(x_train, start_dim=1, end_dim=-1).numpy()
  flattened_x_test = torch.flatten(x_test, start_dim=1, end_dim=-1).numpy()

  # (num_train, num_test) -> [i, j]
  for i in range(num_train):
    train_vector = flattened_x_train[i]

    for j in range(num_test):
      test_vector =  flattened_x_test[j]

      # Use math to calculate Euclidean distance
      # distance = [float((a - b)**2) for a, b in zip(train_vector, test_vector)]
      # distance = math.sqrt(sum(distance))

      # Use numpy to calculate Euclidean distance
      distance = np.sqrt(np.sum((train_vector - test_vector)**2, axis=0))

      # Euclidean distance between the ith training point and jth test point
      dists[i, j] = torch.FloatTensor([distance])
  
  return dists


def compute_distances_one_loop(x_train: torch.Tensor,
                               x_test: torch.Tensor):
  """
    Computes the squared Euclidean distance between each element of the training
    set and each element of the test set. Images should be flattened and treated
    as vectors.

    This implementation uses only a single loop over the training data.

    Similar to compute_distances_two_loops, this should be able to handle inputs
    with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
    You may not use any functions from torch.nn or torch.nn.functional.

    Inputs:
    - x_train: Torch tensor of shape (num_train, D1, D2, ...)
    - x_test: Torch tensor of shape (num_test, D1, D2, ...)

    Returns:
    - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
      squared Euclidean distance between the ith training point and the jth test
      point.
  """

  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]

  dists = x_train.new_zeros(num_train, num_test)

  # Able to handle inputs with any number of dimensions
  flattened_x_train = torch.flatten(x_train, start_dim=1, end_dim=-1).numpy()
  flattened_x_test = torch.flatten(x_test, start_dim=1, end_dim=-1).numpy()

  # (num_train, num_test) -> [i, j]
  for i in range(num_train):
    train_vector = flattened_x_train[i]
    
    distance = np.sqrt(np.sum((train_vector - flattened_x_test)**2, axis=1))
    dists[i, :] = torch.FloatTensor(distance)
  
  return dists


def compute_distances_no_loops(x_train: torch.Tensor,
                               x_test: torch.Tensor):
  """
    Computes the squared Euclidean distance between each element of the training
    set and each element of the test set. Images should be flattened and treated
    as vectors.

    This implementation should not use any Python loops. For memory-efficiency,
    it also should not create any large intermediate tensors; in particular you
    should not create any intermediate tensors with O(num_train*num_test)
    elements.

    Similar to compute_distances_two_loops, this should be able to handle inputs
    with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
    You may not use any functions from torch.nn or torch.nn.functional.
    Inputs:
    - x_train: Torch tensor of shape (num_train, C, H, W)
    - x_test: Torch tensor of shape (num_test, C, H, W)

    Returns:
    - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
      squared Euclidean distance between the ith training point and the jth test
      point.
  """

  # Handle inputs with any number of dimensions
  flattened_x_train = torch.flatten(x_train, start_dim=1, end_dim=-1).numpy()
  flattened_x_test = torch.flatten(x_test, start_dim=1, end_dim=-1).numpy()

  # Compute squared distances between all pairs of training and test examples
  train_squares = np.sum(flattened_x_train ** 2, axis=1, keepdims=True)
  test_squares = np.sum(flattened_x_test ** 2, axis=1, keepdims=True)

  # Take the dot product between list of vectors
  cross = np.dot(flattened_x_train, flattened_x_test.T)
  squared_dists = train_squares + test_squares.T - 2 * cross

  # Take square root to obtain Euclidean distance
  dists = torch.FloatTensor(np.sqrt(squared_dists))

  return dists


def predict_labels(dists: torch.Tensor,
                   y_train: torch.Tensor,
                   k: int=1):
  """
    Given distances between all pairs of training and test samples, predict a
    label for each test sample by taking a **majority vote** among its k nearest
    neighbors in the training set.

    In the event of a tie, this function **should** return the smallest label. For
    example, if k=5 and the 5 nearest neighbors to a test example have labels
    [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes), so
    we should return 1 since it is the smallest label.

    This function should not modify any of its inputs.

    Inputs:
    - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
      squared Euclidean distance between the ith training point and the jth test
      point.
    - y_train: Torch tensor of shape (num_train,) giving labels for all training
      samples. Each label is an integer in the range [0, num_classes - 1]
    - k: The number of nearest neighbors to use for classification.

    Returns:
    - y_pred: A torch int64 tensor of shape (num_test,) giving predicted labels
      for the test data, where y_pred[j] is the predicted label for the jth test
      example. Each label should be an integer in the range [0, num_classes - 1].
  """

  num_train, num_test = dists.shape
  y_pred = torch.zeros(num_test, dtype=torch.int64)

  for j in range(num_test):
      # Create a list of tuples, where each tuple contains a training sample's
      # label and its distance to the jth test sample
      samples_dist = [(y_train[i], dists[i, j]) for i in range(num_train)]

      # Sort the list of tuples by distance in ascending order
      sorted_samples_dist = sorted(samples_dist, key=lambda x: x[1])

      # Get the k nearest neighbors' labels from the sorted list
      closest_y = [x[0] for x in sorted_samples_dist[:k]]

      # Count the number of occurrences of each label
      label_counts = torch.bincount(torch.tensor(closest_y))

      # Find the label with the most occurrences
      # In the event of a tie, find the smallest label among the tied labels
      y_pred[j] = torch.argmax(label_counts)

  return y_pred


class KnnClassifier(object):

  def __init__(self,
               x_train: torch.Tensor,
               y_train: torch.Tensor):
    """
      Create a new K-Nearest Neighbor classifier with the specified training data.
      In the initializer we simply memorize the provided training data.

      Inputs:
      - x_train: Torch tensor of shape (num_train, C, H, W) giving training data
      - y_train: int64 torch tensor of shape (num_train,) giving training labels
    """

    self.x_train = x_train
    self.y_train = y_train

  def predict(self,
              x_test: torch.Tensor,
              k: int=1):
    """
      Make predictions using the classifier.

      Inputs:
      - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
      - k: The number of neighbors to use for predictions

      Returns:
      - y_test_pred: Torch tensor of shape (num_test,) giving predicted labels
        for the test samples.
    """

    # Calculate distances between training samples are test samples
    distances = compute_distances_no_loops(x_train=self.x_train, x_test=x_test)

    # Predicted labels for the test data
    y_test_pred = predict_labels(dists=distances, y_train=self.y_train, k=k)

    return y_test_pred

  def check_accuracy(self,
                     x_test: torch.Tensor,
                     y_test: torch.Tensor,
                     k: int=1,
                     quiet: bool=False):
    """
      Utility method for checking the accuracy of this classifier on test data.
      Returns the accuracy of the classifier on the test data, and also prints a
      message giving the accuracy.

      Inputs:
      - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
      - y_test: int64 torch tensor of shape (num_test,) giving test labels
      - k: The number of neighbors to use for prediction
      - quiet: If True, don't print a message.

      Returns:
      - accuracy: Accuracy of this classifier on the test data, as a percent.
        Python float in the range [0, 100]
    """

    y_test_pred = self.predict(x_test, k=k)
    
    num_samples = x_test.shape[0]
    num_correct = (y_test == y_test_pred).sum().item()
    
    accuracy = 100.0 * num_correct / num_samples
    
    msg = (f'Got {num_correct} / {num_samples} correct; '
           f'accuracy is {accuracy:.2f}%')
    
    if not quiet:
      print(msg)
    
    return accuracy


def knn_cross_validate(x_train: torch.Tensor,
                       y_train: torch.Tensor,
                       num_folds: int=5,
                       k_choices: list=None):
  """
    Perform cross-validation for KnnClassifier.

    Inputs:
    - x_train: Tensor of shape (num_train, C, H, W) giving all training data
    - y_train: int64 tensor of shape (num_train,) giving labels for training data
    - num_folds: Integer giving the number of folds to use
    - k_choices: List of integers giving the values of k to try

    Returns:
    - k_to_accuracies: Dictionary mapping values of k to lists, where
      k_to_accuracies[k][i] is the accuracy on the ith fold of a KnnClassifier
      that uses k nearest neighbors.
  """

  if k_choices is None:
    # Use default values
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

  # Divide the training data into num_folds equally-sized folds
  x_data_folds = torch.chunk(x_train, num_folds, dim=0)
  y_data_folds = torch.chunk(y_train, num_folds, dim=0)

  # A dictionary holding the accuracies for different values of k that we find
  # when running cross-validation. After running cross-validation,
  # k_to_accuracies[k] should be a list of length num_folds giving the different
  # accuracies we found when trying KnnClassifiers that use k neighbors
  k_to_accuracies = {}
  
  # For each value of k, run the KnnClassifier num_folds times. Store the accuracies
  # for all fold and all values of k in the k_to_accuracies dictionary
  for k in k_choices:
    k_to_accuracies[k] = []
    print("\n#k = " + str(k))

    # Use all folds except the i-th fold as training data
    for ith_fold in range(num_folds):

      # Train and test splits with folded dataset
      x_train_fold = torch.cat(x_data_folds[: ith_fold] + x_data_folds[ith_fold + 1 :], dim=0)
      y_train_fold = torch.cat(y_data_folds[: ith_fold] + y_data_folds[ith_fold + 1 :], dim=0)
      x_test_fold = x_data_folds[ith_fold]
      y_test_fold = y_data_folds[ith_fold]

      # Create a K-Nearest Neighbors Classifier object instance
      classifier = KnnClassifier(x_train_fold, y_train_fold)

      accuracy = classifier.check_accuracy(x_test_fold, y_test_fold, k=k)
      k_to_accuracies[k].append(accuracy)
  
  return k_to_accuracies


def knn_get_best_k(k_to_accuracies: dict):
  """
    Select the best value for k, from the cross-validation result from
    knn_cross_validate. If there are multiple k's available, then you SHOULD
    choose the smallest k among all possible answer.

    Inputs:
    - k_to_accuracies: Dictionary mapping values of k to lists, where
      k_to_accuracies[k][i] is the accuracy on the ith fold of a KnnClassifier
      that uses k nearest neighbors.

    Returns:
    - best_k: best (and smallest if there is a conflict) k value based on
              the k_to_accuracies info
  """

  best_k = 0
  best_accuracy = 0.0

  # Find the best k
  for k in k_to_accuracies:
    accuracies = k_to_accuracies[k]
    mean_accuracy = np.mean(accuracies)

    if mean_accuracy > best_accuracy:
      best_accuracy = mean_accuracy
      best_k = k

  return best_k
