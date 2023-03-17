"""
  General utilities to help with implementation
"""
import random
import torch


def reset_seed(number: int):
  """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
  """
  random.seed(number)
  torch.manual_seed(number)

  return
