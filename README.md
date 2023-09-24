# CS231n Spring 2023 - Assignment Answers

[CS231n: Deep Learning for Computer Vision](http://cs231n.stanford.edu/)作为CV领域入门的经典课程，我在看完[课程讲解](https://www.bilibili.com/video/BV1nJ411z7fe?p=1&vd_source=05f97c55a318d0682c7cce673cbb8506)后，按照[Schedul](http://cs231n.stanford.edu/schedule.html)的顺序，在[Course Notes](https://cs231n.github.io/)的帮助下，完成了3次作业（[Assignment 1](https://cs231n.github.io/assignments2023/assignment1/), [Assignment 2](https://cs231n.github.io/assignments2023/assignment2/)  and [Assignment 3](https://cs231n.github.io/assignments2023/assignment3/)），并在此记录作业的解答过程。

注：所有和中文相关的内容、注释、推导过程均为本人所写，如有错误，欢迎指正。


## Assignment1（completed）

### Q1: k-Nearest Neighbor classifier!

The notebook **knn.ipynb** will walk you through implementing the kNN classifier.

### Q2: Training a Support Vector Machine

The notebook **svm.ipynb** will walk you through implementing the SVM classifier.

### Q3: Implement a Softmax classifier

The notebook **softmax.ipynb** will walk you through implementing the Softmax classifier.

### Q4: Two-Layer Neural Network

The notebook **two_layer_net.ipynb** will walk you through the implementation of a two-layer neural network classifier.

### Q5: Higher Level Representations: Image Features

The notebook **features.ipynb** will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.



## Assignment2

### Q1: Multi-Layer Fully Connected Neural Networks

The notebook `FullyConnectedNets.ipynb` will have you implement fully connected networks of arbitrary depth. To optimize these models you will implement several popular update rules.

### Q2: Batch Normalization

In notebook `BatchNormalization.ipynb` you will implement batch normalization, and use it to train deep fully connected networks.

### Q3: Dropout

The notebook `Dropout.ipynb` will help you implement dropout and explore its effects on model generalization.

### Q4: Convolutional Neural Networks

In the notebook `ConvolutionalNetworks.ipynb` you will implement several new layers that are commonly used in convolutional networks.

### Q5: PyTorch on CIFAR-10

For this part, you will be working with PyTorch, a popular and powerful deep learning framework.

Open up `PyTorch.ipynb`. There, you will learn how the framework works, culminating in training a convolutional network of your own design on CIFAR-10 to get the best performance you can.



## Assignment3

### Q1: Network Visualization: Saliency Maps, Class Visualization, and Fooling Images

The notebook `Network_Visualization.ipynb` will introduce the pretrained SqueezeNet model, compute gradients with respect to images, and use them to produce saliency maps and fooling images.

### Q2: Image Captioning with Vanilla RNNs

The notebook `RNN_Captioning.ipynb` will walk you through the implementation of vanilla recurrent neural networks and apply them to image captioning on COCO.

### Q3: Image Captioning with Transformers

The notebook `Transformer_Captioning.ipynb` will walk you through the implementation of a Transformer model and apply it to image captioning on COCO.

### Q4: Generative Adversarial Networks

In the notebook `Generative_Adversarial_Networks.ipynb` you will learn how to generate images that match a training dataset and use these models to improve classifier performance when training on a large amount of unlabeled data and a small amount of labeled data. **When first opening the notebook, go to `Runtime > Change runtime type` and set `Hardware accelerator` to `GPU`.**

### Q5: Self-Supervised Learning for Image Classification

In the notebook `Self_Supervised_Learning.ipynb`, you will learn how to leverage self-supervised pretraining to obtain better performance on image classification tasks. **When first opening the notebook, go to `Runtime > Change runtime type` and set `Hardware accelerator` to `GPU`.**

### Extra Credit: Image Captioning with LSTMs

The notebook `LSTM_Captioning.ipynb` will walk you through the implementation of Long-Short Term Memory (LSTM) RNNs and apply them to image captioning on COCO.



## Reference

[hanlulu1998/CS231n: Stanford University CS231n 2016 winter assignments (github.com)](https://github.com/hanlulu1998/CS231n)

