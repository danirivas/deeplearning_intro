# Concepts
- Ablation Study [[1]](http://qingkaikong.blogspot.com/2017/12/what-is-ablation-study-in-machine.html) [[2]](https://www.quora.com/In-the-context-of-deep-learning-what-is-an-ablation-study)
- The Vanishing Gradient Problem [[1]](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)

# Basics

...Basic concepts to start with Deep Learning, before delving into the state-of-the-art 
- Neural Network [[Video]](https://www.youtube.com/watch?v=aircAruvnKk)
- Gradient Descent [[Video]](https://www.youtube.com/watch?v=IHZwWFHWa-w) [[1]](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
- Backpropagation [[Video]](https://www.youtube.com/watch?v=Ilg3gGewQ5U) 
..- Calculus of backpropagation [[Video]](https://www.youtube.com/watch?v=tIeHLnjs5U8)


# Deep Learning

## Introduction

 - Learning Neural Network Architectures. [[1]](https://towardsdatascience.com/learning-neural-network-architectures-6109cb133caf)

## Neural Network Architectures

[[Here]](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba) you can see a summary of some of the most important neural networks of the last years. [[Article]](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba) (Mar 2017) and [[Update]](https://medium.com/@culurciello/analysis-of-deep-neural-networks-dcf398e71aae) (Sep 2018).

The same authors revise [[here]](https://arxiv.org/pdf/1605.07678.pdf) the different architectures for practical applications.


...Within each category, models are ordered chronologically.

### Image Classification

...Review of Deep Learning Algorithms for Image Classification. [[1]](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-image-classification-5fdbca4a05e2)

- AlexNet (2012) [[Paper]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- VGGNet (2015) [[Paper]](https://arxiv.org/pdf/1409.1556.pdf) [Review](https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11) 
- ResNet (2015) [[Paper]](https://arxiv.org/pdf/1512.03385)
- Squeezenet (2016) [[Paper]](https://arxiv.org/pdf/1602.07360.pdf) [[github]](https://github.com/DeepScale/SqueezeNet)

### Segmentation

- Instance Segmentation:
..- DeepMask [[Paper]](https://arxiv.org/abs/1506.06204) [[Review]](https://towardsdatascience.com/review-deepmask-instance-segmentation-30327a072339) [[Demo]](https://www.youtube.com/watch?v=Lbzgh6NkMMw)
- Semantic Segmentation

### Object Detection

...Review of Deep Learning Algorithms for Object Detection. [[1]](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)

- YOLO (2016) [[Paper]](https://arxiv.org/pdf/1506.02640.pdf)
- YOLO9000, a.k.a. YOLOv2 (2016) [[Paper]](https://arxiv.org/pdf/1612.08242.pdf)
- YOLOv3 (2018) [[Paper]](https://arxiv.org/pdf/1804.02767.pdf)

#### Datasets
- PASCAL Visual Object Classification (PASCAL VOC) [[link]](http://host.robots.ox.ac.uk/pascal/VOC/)
- Common Objects in COntext (COCO)

## Generative Adversial Networks

- Introduction [[link]](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)
- First paper "Generative Adversarial Networks" in 2014 [[Paper]](https://arxiv.org/pdf/1406.2661.pdf)


# Programming

## Frameworks
- Tensorflow (Google)
- PyTorch (Facebook)
- OpenVINO (Intel, only inference)

## Tutorials

- Getting Started with TensorFlow and Deep Learning, from ScyPy 2018 by Josh Gordon (Google) [[link]](https://www.youtube.com/watch?v=tYYVSEHq-io&t=709s)


# Other interesting links

- *What happens when our computers get smarter than we are?* by Nick Bostrom. [[video]](https://www.youtube.com/watch?v=MnT1xgZgkpk)
- *Can we build AI without losing control over it?* by Sam Harris. [[video]](https://www.youtube.com/watch?v=8nt3edWLgIg)

## Interesting YouTube Channels:
- Two Minute Papers. [[link]](https://www.youtube.com/user/keeroyz/)


# Terminology

- What is a *model* and what is an *architecture*?
```
This terminology is often abused, but the pedantic view is:

A model would be a network architecture with all it's weights viewed as free parameters.
A fit model is a network with fixed weights determined by running a fitting algorithm with some training data.
Parameters map out the various specific shapes that the model can obtain, fitting chooses specific values of the weights that best reflect the training data.
Hyperparameters control the behaviour of the fitting algorithm, they are often set to find the parameters that offer the best performance according to some estimate of hold-out error.
I settled on this terminology after reading Wasserman's All of Statistics.

It's very common to call the fit model just a model. I try to use my words precisely and consistently, especially when talking to students, but it is hard to avoid sometimes!
```
[[link to stackoverflow answer]](https://stats.stackexchange.com/a/291482)


