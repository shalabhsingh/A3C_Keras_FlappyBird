# A3C_Keras_FlappyBird
![](animation.gif)

This repository uses Asynchronous advantage actor-critic algorithm (A3C) to play Flappy Bird using Keras. The details of this algorithm are mentioned in [this paper](https://arxiv.org/pdf/1602.01783.pdf) by Deep Mind.

# What is A3C ?
A3C refers to Asynchronous advantage actor-critic algorithm for deep reinforcement learning. It was proposed over the famous DQN network for playing atari games, first made by DeepMind back in 2013. DQN was the first RL algorithm which was able to play games successfully because of which Google bought DeepMind itself. However it had some drawbacks which have been solved by A3C-

* DQN had a very large training time (~1 week on a GPU) whereas basic A3C takes 1 day to train on a CPU. (infact training time for Flappy Bird game in this project was just 6 hours !!)
* DQN used experience replay for getting good convergence which requires a lot of memory. A3C use multiple threads for this purpose which eliminates huge memory requirement.
* The major objective of DQN is to estimate Q-value for all actions in different environment states possible. As a result in early stages of learning, it mostly tries to learn Q-value for states which won't even be a part of optimal strategy. On the other hand, A3C learns the best policy to take at good states at the current point of time because of which it is faster to train.

However because of better exploration by DQN, it generally settles at global minima whereas A3C might settle at a local minima leading to sub-optimal results.

**Learning Resources-**

1. For theoretical and implementation details of how a DQN works, see this blog page- https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
2. For theoretical and implementation details of how an A3C works, see this blog page- https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/

# Model Desciption
The input to the neural network is a stack of 4, 84x84 grayscale game frames. The network used a convolutional layer with 16 filters of size 8x8 with stride 4, followed by a convolutional layer with with 32 filters of size 4x4 with stride 2, followed by a fully connected layer with 256 hidden units. All three hidden layers were followed by a rectifier nonlinearity. There are two set of outputs – a softmax output with one output representing the probability of flapping, and a single linear output representing the value function.

The hyperparameters used are-
* Learning rate = 7e-4 which is decreased by 3.2e-8 (can be tuned better) every update.
* No. of threads = 16
* Frames/thread used for each update = 5
* Reward discount (gamma) = 0.99
* RMSProp cache decay rate = 0.99
* Entropy regularization, as suggested by the paper has not been used. However I believe that using it, could lead to better performance of the model.

The best model I have got is still not very good but is able to cross 5 pipes on average (i.e. it has developed a good understanding of when to flap and when not to). To train better models, tinkering with above hyperparameters can be beneficial.

# Installation Dependencies
* Python 3.5
* Keras 2.0
* pygame 
* scikit-image

# How to Run?
Clone the repository or download it. To test the pretrained model, run the "test.py" file. To retrain the model from scratch, run the "train_network.py" file and the trained models at different stages will be saved in "saved_models" folder.

# Disclaimer
This work is based on the following repos and blogs-

1. https://github.com/yenchenlin/DeepLearningFlappyBird
2. https://github.com/jaara/AI-blog
3. http://karpathy.github.io/2016/05/31/rl/

