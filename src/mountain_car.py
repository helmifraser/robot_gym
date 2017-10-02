#!/usr/bin/env python

import gym
import time
import numpy
import random
import time
import tensorflow as tf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
# hyperparams
mountain_car_pos_thresh = [-1.2, 0.6]
mountain_car_vel_thresh = [-0.07, 0.07]
max_episodes = 50

highscore = -2000

#Initialize table with all zeros
Q = np.zeros([2, 3])
# Set learning parameters
learning_rate = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # env.render()

        #Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state,:] + np.random.randn(1,3)*(1./(i+1)))
        #Get new state and reward from environment
        s1,reward,d,_ = env.step(action)
        #Update Q-Table with new knowledge
        # Bellman equation: expected long-term reward for a given action is equal
        # to the immediate reward from the current action combined with the expected
        # reward from the best future action taken at the following state
        Q[state,action] = Q[state,action] + learning_rate*(reward + y*np.max(Q[s1,:]) - Q[state,action])
        rAll += reward
        state = s1
        if done == True:
            break
    #jList.append(j)
    rList.append(rAll)

print "Score over time: " +  str(sum(rList)/num_episodes)

print "Final Q-Table Values"
print Q

# # Parameters
# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# display_step = 1
#
# # Network Parameters
# n_hidden_1 = 20 # 1st layer number of neurons
# n_hidden_2 = 20 # 2nd layer number of neurons
# n_input = 2 # input (position, velocity)
# n_classes = 3 # classes, discrete car movements (0, 1, 2)
#
# # tf Graph input
# X = tf.placeholder("float", [None, n_input])
# Y = tf.placeholder("float", [None, n_classes])
#
# # Store layers weight & bias
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
#
# # Create model
# def multilayer_perceptron(x):
#     # Hidden fully connected layer with n_hidden neurons
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     # Hidden fully connected layer with n_hidden neurons
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     return out_layer
#
# # Construct model
# logits = multilayer_perceptron(X)
#
# # Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
# # Initializing the variables
# init = tf.global_variables_initializer()
#
# # Initialize Session and run `result`
# with tf.Session() as sess:
#   output = sess.run(result)
#   print(output)

# for episode in range(0, max_episodes):
#     observation = env.reset()
#     points = 0
#     done = False
#     while not done:
#         env.render()
#         if observation[0] < 0:
#             action = 2
#         elif observation[0] > 0:
#             action = 0
#         observation, reward, done, info = env.step(action)
#         points += reward
#         if done is True:
#             if points > highscore:      # record high score
#                 highscore = points
#             print("Score: {}, {}".format(highscore, points))
#             break
