from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

# 数值计算损失和梯度


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength -> lambda

    Returns a tuple of:  返回损失(小数)和梯度(数组矩阵)
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero. [D, C]

    # compute the loss and the gradient
    num_classes = W.shape[1]  # 类别数 C
    num_train = X.shape[0]  # 训练样本数 N
    delta = 1.0  # SVM loss function中的delta
    for i in range(num_train):
        scores = X[i].dot(W)  # [1, D] * [D, C] = [1, C]
        correct_class_score = scores[y[i]]  # 选择i样本真实类别的预测分数
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta  # delta = 1
            if margin > 0:
                loss += margin  # SVM loss formula
                # 梯度可参考svm.ipynb中的损失函数微分公式
                dW[:, j] += X[i]  # 对于j类别的梯度
                dW[:, y[i]] -= X[i]  # 对于真实类别的梯度

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)  # 加入正则项，防止过拟合(借助于L2范数，类似奥卡姆剃刀原理——越简单越好)
    dW += reg * 2 * W  # 正则项的偏导数，修改梯度

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


# 解析计算损失和梯度
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    delta = 1.0

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pass
    loss = 0.0
    num_train = X.shape[0]  # 训练样本数 N

    scores = X.dot(W)  # [N, D] * [D, C] = [N, C]
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)  # 取每个样本真实类别的得分，并转为[N, 1]
    margins = scores - correct_class_scores + delta  # [N, C]
    margins[np.arange(num_train), y] = 0  # 将真实类别的margin置为0
    margins = np.maximum(0, margins)
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pass
    # 计算W的梯度
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    margins[margins > 0] = 1  # 将大于0的margin置为1，用作指示性函数. [N, C]
    # 计算每个样本有多少个margin大于0. [N, 1]
    positive_num_classes = np.sum(margins, axis=1)
    # 将真实类别的梯度置为负值的样本数 (而其他类别设为+1, 不用管)
    margins[np.arange(num_train), y] = -positive_num_classes
    dW = X.T.dot(margins)  # 乘以X的转置，得到梯度. [D, N] * [N, C] = [D, C]
    dW /= num_train
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
