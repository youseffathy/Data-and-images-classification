import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)    # For numerical stability
    loss += - np.log(np.exp(scores[y[i]]) / np.sum(np.exp(scores)))
    for j in range(num_classes):
      dW[:, j] += (np.exp(scores[j]) * X[i]) / np.sum(np.exp(scores))
    dW[:, y[i]] -= X[i]

  # Calculate the mean
  loss /= num_train
  dW /= num_train

  # Add the regularization to the loss and gradient.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_train = X.shape[0]

    # Compute the scores and loss
    scores = X.dot(W)
    scores -= np.expand_dims(np.amax(scores, axis=1), axis=1)   # For numerical stability
    scores = np.exp(scores)
    correct_score = np.expand_dims(scores[np.arange(num_train), y], axis=1)
    scores_sum = np.expand_dims(np.sum(scores, axis=1),axis=1)
    loss = -np.sum(np.log(correct_score/scores_sum))

    # Compute the gradient
    derivate = scores / scores_sum
    derivate[np.arange(num_train), y] -= 1
    derivate /= num_train
    dW = X.T.dot(derivate)

    # Mean
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W


    return loss, dW

