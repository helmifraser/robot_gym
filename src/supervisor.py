import numpy as np
import math
import os
import time

import turtlebot, mlp, genetic_algorithm
import rospy
import std_srvs.srv, gazebo_msgs.srv

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
            rospy.loginfo("Supervisor: Service call (reset_world) failed: %s"%e)

    def pause_sim_physics(self):
        rospy.logwarn("Supervisor: Waiting for pause physics service...")
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pause = rospy.ServiceProxy("/gazebo/pause_physics", std_srvs.srv.Empty())
            pause()
            rospy.loginfo("Supervisor: Pausing physics")
        except rospy.ServiceException, e:
            rospy.loginfo("Supervisor: Service call (pause_physics) failed: %s"%e)

    def set_physics_properties(self, properties):
        rospy.logwarn("Supervisor: Waiting for physics properties service...")
        rospy.wait_for_service("/gazebo/set_physics_properties")
        time_step = properties.time_step
        max_update_rate = properties.max_update_rate
        gravity = properties.gravity
        ode_config = properties.ode_config
        # print("{}, {}, {}, {}".format(time_step, max_update_rate, gravity, ode_config))
        try:
            set = rospy.ServiceProxy("/gazebo/set_physics_properties", gazebo_msgs.srv.SetPhysicsProperties)
            set(time_step, max_update_rate, gravity, ode_config)
            rospy.loginfo("Supervisor: Updating physics properties")
        except rospy.ServiceException, e:
            rospy.loginfo("Supervisor: Service call (set_physics_properties) failed: %s"%e)

    def get_physics_properties(self):
        rospy.logwarn("Supervisor: Waiting for physics properties service...")
        rospy.wait_for_service("/gazebo/get_physics_properties")
        try:
            get = rospy.ServiceProxy("/gazebo/get_physics_properties", gazebo_msgs.srv.GetPhysicsProperties())
            rospy.loginfo("Supervisor: Getting physics properties")
            return get()
        except rospy.ServiceException, e:
            rospy.loginfo("Supervisor: Service call (get_physics_properties) failed: %s"%e)
            return 0

    def update_time_step(self, new_time_step):
        current_properties = self.get_physics_properties()
        properties = self.convert_to_set(current_properties)

        properties.time_step = new_time_step
        # print("update: \n {},\n {},\n {},\n {}".format(properties.time_step, properties.max_update_rate, properties.gravity, properties.ode_config))
        self.set_physics_properties(properties)
        print("update_time_step: Updated time_step to {}".format(new_time_step))

    def convert_to_set(self, current_properties):
        properties = gazebo_msgs.srv.SetPhysicsProperties()
        properties.time_step = current_properties.time_step
        properties.max_update_rate = current_properties.max_update_rate
        properties.gravity = current_properties.gravity
        properties.ode_config = current_properties.ode_config
        # print("convert: \n {},\n {},\n {},\n {}".format(properties.time_step, properties.max_update_rate, properties.gravity, properties.ode_config))
        return properties

    def return_rate(self):
        return self.rate

def main():
    supervisor = Supervisor()
    node_rate = supervisor.return_rate()

    # supervisor.reset_world()
    supervisor.update_time_step(0.1)
    while rospy.is_shutdown() is not True:
        node_rate.sleep()

if __name__ == "__main__":
    main()
