#!/usr/bin/env python

import gym
import gym_gazebo
import time
import numpy
import random
import time
import rospy
import os
import signal
from rosgraph_msgs.msg import Log
from std_msgs.msg import Bool

import matplotlib
import matplotlib.pyplot as plt

import mlp

class TurtlebotLearning(object):
    """Reinforcement learning of a turtlebot using Gazebo and OpenAI Gym"""

    def __init__(self):
        super(TurtlebotLearning, self).__init__()
        self.env = gym.make('GazeboCircuitTurtlebotLidar-v0')
        rospy.on_shutdown(self.shutdown_hook)
        # rospy.init_node('turtlebot_learn')
        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)
        self.validation = "empty"
        self.rosout_sub = rospy.Subscriber('/rosout', Log, self.rosout_cb)
        self.reset_sub = rospy.Subscriber('/reset_env', Bool, self.reset_env_cb)
        self.pid = os.getpid()

        self.wait_for_gym()
        self.observation, self.reward, self.done, self.info = self.env.reset()
        self.controller = mlp.MLP_NeuralNetwork(input=8,hidden=16,output=2)

    def shutdown_hook(self):
        """
            Callback function invoked when initiating a node shutdown. Called
            before actual shutdown occurs.
        """

        rospy.logwarn("Shutting down all the things fam")
        # os.kill(self.pid, signal.SIGKILL)

    def wait_for_gym(self):
        """
            I know, I know it's so hacky it's disgusting
        """

        completed = False
        while completed is False:
            if "GazeboRosKobuki" in self.validation:
                completed = True
                rospy.logwarn("Successfully initialised, ready to learn! Run 'gzclient' in another terminal")

    def rosout_cb(self, msg):
        self.validation = msg.msg

    def reset_and_update(self):
        self.observation = self.env.reset()

    def reset_env_cb(self, msg):
        if msg.data == True:
            # self.env.reset()
            self.reset_and_update()
            print("Observation: {}".format(self.return_observation()))

    def render_env(self):
        """
            This causes Gazebo to pop up, don't use it
        """
        self.env.render()

    def return_observation(self):
        return self.observation

    def return_rate(self):
        return self.rate

    def return_mlp(self):
        return self.controller

    def test(self):
        print("testing")

def main():
    try:
        instance = TurtlebotLearning()
        node_rate = instance.return_rate()
        mlp_controller = instance.return_mlp()
        wi = np.random.randn(8, 16)
        wo = np.random.randn(16, 2)

        while rospy.is_shutdown() is not True:
            # instance.test()
            node_rate.sleep()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
