# Where to start?

- A Beginner's Guide To Understanding Convolutional Neural Networks. [[Part1]](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/) [[Part2]](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

# Concepts
- Ablation Study [[1]](http://qingkaikong.blogspot.com/2017/12/what-is-ablation-study-in-machine.html) [[2]](https://www.quora.com/In-the-context-of-deep-learning-what-is-an-ablation-study)
- The Vanishing Gradient Problem [[1]](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)

# Basics

Basic concepts to start with Deep Learning, before delving into the state-of-the-art 
- Neural Network [[Video]](https://www.youtube.com/watch?v=aircAruvnKk)
- Gradient Descent [[Video]](https://www.youtube.com/watch?v=IHZwWFHWa-w) [[1]](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
- Backpropagation [[Video]](https://www.youtube.com/watch?v=Ilg3gGewQ5U) 
- Calculus of backpropagation [[Video]](https://www.youtube.com/watch?v=tIeHLnjs5U8)


## Layers

### Activation Functions

- ReLU: Rectified Linear Units Improve Restricted Boltzmann Machines (2010) [[paper]](http://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)

```
[...]After some ReLU layers, programmers may choose to apply a pooling layer. It is also referred to as a downsampling layer. In this category, there are also several layer options, with maxpooling being the most popular. This basically takes a filter (normally of size 2x2) and a stride of the same length. It then applies it to the input volume and outputs the maximum number in every subregion that the filter convolves around.
```
(https://adeshpande3.github.io/assets/MaxPool.png "Example of MaxPool")
[[source]](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

```
 The intuitive reasoning behind this layer is that once we know that a specific feature is in the original input volume (there will be a high activation value), its exact location is not as important as its relative location to the other features. As you can imagine, this layer drastically reduces the spatial dimension (the length and the width change but not the depth) of the input volume. This serves two main purposes. The first is that the amount of parameters or weights is reduced by 75%, thus lessening the computation cost. The second is that it will control overfitting.
```

### Dropout

```
The idea of dropout is simplistic in nature. This layer “drops out” a random set of activations in that layer by setting them to zero. Simple as that. Now, what are the benefits of such a simple and seemingly unnecessary and counterintuitive process? Well, in a way, it forces the network to be redundant. By that I mean the network should be able to provide the right classification or output for a specific example even if some of the activations are dropped out. It makes sure that the network isn’t getting too “fitted” to the training data and thus helps alleviate the overfitting problem. An important note is that this layer is only used during training, and not during test time.
```
[[source]](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

See also: *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (2014) [[paper]](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf))

### Network In Network Layer

```
A network in network layer refers to a conv layer where a 1 x 1 size filter is used. Now, at first look, you might wonder why this type of layer would even be helpful since receptive fields are normally larger than the space they map to. However, we must remember that these 1x1 convolutions span a certain depth, so we can think of it as a 1 x 1 x N convolution where N is the number of filters applied in the layer. Effectively, this layer is performing a N-D element-wise multiplication where N is the depth of the input volume into the layer.
```

See also: *Network In Network* by Min Li. [[arxiv.org]](https://arxiv.org/pdf/1312.4400v3.pdf)

# Deep Learning

## Introduction

 - Learning Neural Network Architectures. [[1]](https://towardsdatascience.com/learning-neural-network-architectures-6109cb133caf)

## Neural Network Architectures

[[Here]](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba) you can see a summary of some of the most important neural networks of the last years. [[Article]](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba) (Mar 2017) and [[Update]](https://medium.com/@culurciello/analysis-of-deep-neural-networks-dcf398e71aae) (Sep 2018).

The same authors revise [[here]](https://arxiv.org/pdf/1605.07678.pdf) the different architectures for practical applications.


Within each category, models are ordered chronologically.

### Image Classification

Review of Deep Learning Algorithms for Image Classification. [[1]](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-image-classification-5fdbca4a05e2)

- AlexNet (2012) [[Paper]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- VGGNet (2015) [[Paper]](https://arxiv.org/pdf/1409.1556.pdf) [Review](https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11) 
- ResNet (2015) [[Paper]](https://arxiv.org/pdf/1512.03385)
- Squeezenet (2016) [[Paper]](https://arxiv.org/pdf/1602.07360.pdf) [[github]](https://github.com/DeepScale/SqueezeNet)

### Object Detection

Review of Deep Learning Algorithms for Object Detection. [[1]](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)
Performance analysis (speed and accuracy) comparison between different object detection architectures. [[1]](https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359)

- YOLO (2016) [[Paper]](https://arxiv.org/pdf/1506.02640.pdf)
- YOLO9000, a.k.a. YOLOv2 (2016) [[Paper]](https://arxiv.org/pdf/1612.08242.pdf)
- YOLOv3 (2018) [[Paper]](https://arxiv.org/pdf/1804.02767.pdf)

- *Real-time Object Detection with YOLO, YOLOv2 and now YOLOv3

- Single Shot Detector (2016). Also known as SSD. [[paper]](https://arxiv.org/abs/1512.02325) [[review]](https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11)

*What do we learn from region-based object detectors (Faster R-CNN, R-FCN, FPN)?* [[Part 1]](https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9)
*What do we learn from single shot object detectors (SSD, YOLOv3), FPN & Focal loss (RetinaNet)? [[Part 2]](https://medium.com/@jonathan_hui/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d)

#### Segmentation

##### Instance Segmentation
- DeepMask [[Paper]](https://arxiv.org/abs/1506.06204) [[Review]](https://towardsdatascience.com/review-deepmask-instance-segmentation-30327a072339) [[Demo]](https://www.youtube.com/watch?v=Lbzgh6NkMMw)
##### Semantic Segmentation

## Datasets
- PASCAL Visual Object Classification (PASCAL VOC) [[link]](http://host.robots.ox.ac.uk/pascal/VOC/)
- Common Objects in COntext (COCO)

## Architectures

- Network In Network (2013) [[arxiv.org]](https://arxiv.org/pdf/1312.4400.pdf)
- Inception (2014) [[arxiv.org]](https://arxiv.org/pdf/1409.4842.pdf)


## Generative Adversial Networks

- Introduction [[link]](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)
- First paper "Generative Adversarial Networks" in 2014 [[Paper]](https://arxiv.org/pdf/1406.2661.pdf)

- *How deep learning fakes videos (Deepfake) and how to detect it?* [[link]](https://medium.com/@jonathan_hui/how-deep-learning-fakes-videos-deepfakes-and-how-to-detect-it-c0b50fbf7cb9)


# Visualization and Representation of Neural Networks

- *Visualizing and Understanding Convolutional Networks* (2013) [[arxiv.org]](https://arxiv.org/pdf/1311.2901.pdf)
- *Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images* (2015) [[CV Foundation]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)
- *Understanding Neural Networks Through Deep Visualization* (2015) [[arxiv.org]](https://arxiv.org/pdf/1311.2901.pdf) [[website]](http://yosinski.com/deepvis) [[code]](https://github.com/yosinski/deep-visualization-toolbox) [[video]](https://youtu.be/AgkfIQ4IGaM)

- Inceptionism: *Going Deeper into Neural Networks* [[Google AI Blog]](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) 

- *How computers are learning to be creative*, by Blaise Agüera y Arcas [[video]](https://www.youtube.com/watch?v=uSUOdu_5MPc)

# Programming

## Frameworks
- Tensorflow (Google)
- PyTorch (Facebook)
- OpenVINO (Intel, only inference)

- [OpenAI Gym](https://gym.openai.com). A toolkit for developing and comparing reinforcement learning algorithms.
- [OpenAI Universe](https://openai.com/blog/universe/). "With Universe, any program can be turned into a Gym environment."


## Tutorials

- Getting Started with TensorFlow and Deep Learning, from ScyPy 2018 by Josh Gordon (Google) [[link]](https://www.youtube.com/watch?v=tYYVSEHq-io&t=709s)


# Courses

- [fast.ai](https://www.fast.ai/) - [Course Overview](https://towardsdatascience.com/simplifying-deep-learning-with-fast-ai-37aa0d321f5e)
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) - Google

# Books

- [deeplearningbook.org](https://deeplearningbook.org)


# Other interesting links

- *How computers are learning to be creative* by Blaise Agüera y Arcas. [[video]](https://www.youtube.com/watch?v=uSUOdu_5MPc)
- *What happens when our computers get smarter than we are?* by Nick Bostrom. [[video]](https://www.youtube.com/watch?v=MnT1xgZgkpk)
- *Can we build AI without losing control over it?* by Sam Harris. [[video]](https://www.youtube.com/watch?v=8nt3edWLgIg)
- *How computers learn to recognize objects instantly* by Joseph Redmon (YOLO Team). [[video]](https://www.youtube.com/watch?v=Cgxsv1riJhI)
- *The incredible inventions of intuitive AI* by Maurice Conti. [[video]](https://www.youtube.com/watch?v=aR5N2Jl8k14)

## Interesting YouTube Channels:
- Two Minute Papers. [[link]](https://www.youtube.com/user/keeroyz/)
- DotCSV (in Spanish) [[link]](https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg)

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

# TODO:

- IMPORTANT: Use proper citation style.
