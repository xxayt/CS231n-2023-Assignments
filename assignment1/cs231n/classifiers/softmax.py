from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

# 数值计算损失和梯度


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pass
    # compute the loss and the gradient
    num_classes = W.shape[1]  # 类别数 C
    num_train = X.shape[0]
    for i in range(num_train):
        f = X[i].dot(W)  # [1, D] * [D, C] = [1, C]
        f -= np.max(f)  # 为了在除以大数值时保证数值稳定性，使用归一化技巧。减去最大值，等价于f的值平移，使得最大值为0
        loss += np.log(np.sum(np.exp(f))) - f[y[i]]  # softmax loss 公式
        dW[:, y[i]] -= X[i]  # 对于真实类别的梯度
        sum = np.sum(np.exp(f))
        for j in range(num_classes):
            dW[:, j] += np.exp(f[j]) / sum * X[i]  # 对于j类别的梯度

    # 取均值
    loss /= num_train
    dW /= num_train

    # 加入正则项
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


# 解析计算损失和梯度
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pass
    num_train = X.shape[0]  # N
    # 损失 loss
    fs = X.dot(W)  # [N, D] * [D, C] = [N, C]
    fs = fs - np.max(fs, axis=1, keepdims=True)  # 每行减去最大值，保证数值稳定性
    sum = np.sum(np.exp(fs), axis=1, keepdims=True)  # [N, 1]
    loss = np.sum(np.log(sum)) - np.sum(fs[np.arange(num_train), y])  # softmax loss 公式
    # 注意不能写成 loss = np.sum(np.log(sum) - fs[np.arange(num_train), y])
    # 因为np.log(sum): [500, 1]; fs[np.arange(num_train), y]: [500,]
    # 而由于自动广播 np.log(sum) - fs[np.arange(num_train), y]: [500, 500]
    # 所以需要先求和，再减去fs[np.arange(num_train), y]的和

    # 梯度 gradient
    dW = np.exp(fs) / sum  # [N, C] / [N, 1] = [N, C]
    dW[np.arange(num_train), y] -= 1  # 对于真实类别的梯度
    dW = X.T.dot(dW)  # [D, N] * [N, C] = [D, C]

    # 取均值, 加入正则项
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
