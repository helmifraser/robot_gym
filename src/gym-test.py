#!/usr/bin/env python

import gym
import gym_gazebo
import time
import numpy
import random
import time
import rospy

import matplotlib
import matplotlib.pyplot as plt

gym_env = ['CartPole-v0', 'MountainCar-v0']

env = gym.make(gym_env[0])

highscore = 0

for i_episode in range(20):     # run 20 episodes
    observation = env.reset()
    points = 0      # keep track of the reward each episode
    while True:     # run until episode is done
        env.render()
        action = 1 if observation[2] > 0 else 0     # if angle if positive, move right. if angle is negative, move left
        observation, reward, done, info = env.step(action)
        points += reward
        if done is True:
            if points > highscore:      # record high score
                highscore = points
            print("Score: {}, {}".format(highscore, points))
            break
