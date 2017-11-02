import numpy as np
import math
import os
import time

import turtlebot, mlp, genetic_algorithm
import rospy
import std_srvs.srv

class Supervisor(object):
    """Gazebo simulation supervisor for neuroevolution"""

    def __init__(self):
        super(Supervisor, self).__init__()
        rospy.on_shutdown(self.shutdown_hook)
        rospy.init_node('neuroevolution')

        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)

        # self.cmd_vel_pub_ = rospy.Publisher('/mobile_base/commands/velocity', geometry_msgs.msg.Twist, queue_size=1)
        #
        # self.laser_scan_sub_ = rospy.Subscriber('/laserscan', sensor_msgs.msg.LaserScan, self.scan_cb)


    def shutdown_hook(self):
        """
            Callback function invoked when initiating a node shutdown. Called
            before actual shutdown occurs.
        """

        rospy.logwarn("Supervisor: Shutting down simulation")
        os.system('kill %d' % os.getpid())

    def reset_world(self):
        rospy.logwarn("Supervisor: Waiting for reset service...")
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            reset = rospy.ServiceProxy("/gazebo/reset_world", std_srvs.srv.Empty())
            reset()
            rospy.loginfo("Supervisor: Resetting world")
        except rospy.ServiceException, e:
            rospy.loginfo("Supervisor: Service call failed: %s"%e)

    def pause_sim_physics(self):
        rospy.logwarn("Supervisor: Waiting for physics service...")
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            reset = rospy.ServiceProxy("/gazebo/pause_physics", std_srvs.srv.Empty())
            reset()
            rospy.loginfo("Supervisor: Pausing physics")
        except rospy.ServiceException, e:
            rospy.loginfo("Supervisor: Service call failed: %s"%e)
            
    def return_rate(self):
        return self.rate

def main():
    supervisor = Supervisor()
    node_rate = supervisor.return_rate()

    supervisor.reset_world()
    while rospy.is_shutdown() is not True:
        node_rate.sleep()

if __name__ == "__main__":
    main()
