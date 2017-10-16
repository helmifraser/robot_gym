#!/usr/bin/env python

# import gym
# import gym_gazebo
import time
import numpy as np

import rospy
import std_msgs.msg
import geometry_msgs.msg
import mlp

class TurtlebotLearning(object):
    """Reinforcement learning of a turtlebot using Gazebo and OpenAI Gym"""

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(TurtlebotLearning, self).__init__()
        rospy.on_shutdown(self.shutdown_hook)
        rospy.init_node('turtlebot_learn')

        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)
        self.cmd_vel = rospy.Publisher('/mobile_base/commands/velocity', geometry_msgs.msg.Twist, queue_size=10)

        self.controller = mlp.MLP_NeuralNetwork(input=input_nodes,hidden=hidden_nodes,output=output_nodes)
        rospy.loginfo("Created NN")

    def shutdown_hook(self):
        """
            Callback function invoked when initiating a node shutdown. Called
            before actual shutdown occurs.
        """

        rospy.logwarn("Shutting down all the things fam")
        # os.kill(self.pid, signal.SIGKILL)

    def create_zeroed_twist(self):
        msg = geometry_msgs.msg.Twist()
        msg.linear.x = 0
        msg.linear.y = 0
        msg.linear.z = 0

        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0
        return msg

    def publish_vel(self, twist_msg):
        self.cmd_vel.publish(twist_msg)

    def return_rate(self):
        return self.rate

    def return_mlp(self):
        return self.controller

    def test(self):
        print("testing")

def main():
    try:
        input_size = 8
        hidden_size = 16
        output_size = 2
        instance = TurtlebotLearning(input_size, hidden_size, output_size)
        node_rate = instance.return_rate()
        mlp_controller = instance.return_mlp()

        nn_inputs = np.ones(input_size)

        wi = np.random.randn(input_size, hidden_size)
        wo = np.random.randn(hidden_size, output_size)

        twist_msg = instance.create_zeroed_twist()

        commands = mlp_controller.feedForward(nn_inputs)
        # print(commands)
        twist_msg.linear.x = commands[0]
        twist_msg.angular.z = commands[1]

        while rospy.is_shutdown() is not True:
            # instance.test()
            instance.publish_vel(twist_msg)
            node_rate.sleep()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
