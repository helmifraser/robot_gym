import numpy as np
import math
import os
import time

import turtlebot, mlp, genetic_algorithm
import rospy
import std_srvs.srv, gazebo_msgs.srv, gazebo_msgs.msg
import rosgraph_msgs.msg, std_msgs.msg

default = object()

class Supervisor(object):
    """Gazebo simulation supervisor for neuroevolution"""

    def __init__(self):
        super(Supervisor, self).__init__()
        rospy.on_shutdown(self.shutdown_hook)
        rospy.init_node('neuroevolution')

        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)
        self.tracker = rospy.Publisher('/neuroevolution/progress', std_msgs.msg.String, queue_size=1)
        self.clock_sub = rospy.Subscriber('/clock', rosgraph_msgs.msg.Clock, self.sim_clock_cb)
        self.sim_time = 0

        self.default_time_step = 0.01
        self.evaluation_time = 60*5
        self.end_condition = False

        self.network_dimensions = [7, 16, 2]
        rospy.loginfo("Supervisor: Supervisor initialised")
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

    def reset_timer(self, event):
        # rospy.loginfo("Timer called at {}".format(0.000000001*int(str(event.current_real))))
        self.end_condition = True

    def sim_clock_cb(self, msg):
        self.sim_time = msg

    def reset_world(self):
        # rospy.logwarn("Supervisor: Waiting for reset service...")
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            reset = rospy.ServiceProxy("/gazebo/reset_world", std_srvs.srv.Empty())
            reset()
            # rospy.loginfo("Supervisor: Resetting world")
        except rospy.ServiceException, e:
            rospy.loginfo("Supervisor: Service call (reset_world) failed: %s"%e)

    # def reset_world(self):
    #     rospy.logwarn("Supervisor: Waiting for reset service...")
    #     rospy.wait_for_service("/gazebo/set_model_state")
    #     try:
    #         reset = rospy.ServiceProxy("/gazebo/set_model_state", gazebo_msgs.srv.SetModelState)
    #         setmodelstate = gazebo_msgs.srv.SetModelState()
    #         modelstate = gazebo_msgs.msg.ModelState()
    #         modelstate.model_name = "mobile_base"
    #         print(modelstate)
    #         # setmodelstate.request.model_state = modelstate
    #         reset(modelstate)
    #         rospy.loginfo("Supervisor: Resetting world")
    #     except rospy.ServiceException, e:
    #         rospy.loginfo("Supervisor: Service call (reset_world) failed: %s"%e)

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

    def save_top_individuals(self, generation, best=5, filename="best-"):
        for i in range(0, best):
            filename_wi = filename + str(i) + "-wi"
            filename_wo = filename + str(i) + "-wo"

            np.save(filename_wi, generation[i][0])
            np.save(filename_wo, generation[i][1])

    def return_rate(self):
        return self.rate

    def return_network_dimensions(self,):
        return self.network_dimensions

    def return_default_time_step(self):
        return self.default_time_step

    def check_end_condition(self):
        return self.end_condition

    def reset_end_condition(self):
        self.end_condition = False

    def return_evaluation_time(self):
        return self.evaluation_time

    def update_progress(self, progress):
        msg = "Running gen {} individual {}".format(progress[0], progress[1])
        self.tracker.publish(msg)

def test():
    supervisor = Supervisor()
    node_rate = supervisor.return_rate()
    network_dimensions = supervisor.return_network_dimensions()
    time_step = supervisor.return_default_time_step()

    controller = "best-1"

    # Instantiate controller, NN and GA for given dimensions
    robot = turtlebot.TurtlebotController()
    neural_network = mlp.MLP_NeuralNetwork(network_dimensions[0], network_dimensions[1], network_dimensions[2])
    neural_network.load_weights(controller)

    twist_msg = robot.create_zeroed_twist()

    rospy.loginfo("Supervisor: Using controller {}".format(controller))

    while rospy.is_shutdown() is False:
        # Get current laser data
        laser_data = robot.return_segmented_laser_data()

        # Feed forward through network and get next actions
        action = neural_network.feed_forward(laser_data)
        # rospy.loginfo("laser_data: {}\n action: {}".format(laser_data, action))

        # Send actions to 'bot
        twist_msg.linear.x = action[0]
        twist_msg.angular.z = action[1]
        robot.publish_vel(twist_msg)
        node_rate.sleep()

def main():
    # Instantiate supervisor
    supervisor = Supervisor()
    node_rate = supervisor.return_rate()
    network_dimensions = supervisor.return_network_dimensions()
    time_step = supervisor.return_default_time_step()
    evaluation_time = supervisor.return_evaluation_time()

    # Instantiate controller, NN and GA for given dimensions
    robot = turtlebot.TurtlebotController()
    neural_network = mlp.MLP_NeuralNetwork(network_dimensions[0], network_dimensions[1], network_dimensions[2])
    ga = genetic_algorithm.GeneticAlgorithm(network_dimensions)

    # Generate initial population and required params
    initial_generation = ga.initialise_population()
    next_generation = None
    generation_count = 0
    generation_fitness = np.zeros(len(initial_generation))

    # Set sim time step to go faster
    supervisor.update_time_step(time_step)

    # Countdown timer
    rospy.loginfo("Starting in...")
    time.sleep(1)
    for i in range(5, 0, -1):
        rospy.loginfo("{}...".format(i))
        time.sleep(1)
    rospy.loginfo("Punch it.")


    # Go until max no. of generations have been reached or ROS fails
    while generation_count < ga.return_max_gen() and rospy.is_shutdown() is not True:
        # Set initial pop or overwrite with new gen from last iteration
        if generation_count == 0:
            population = initial_generation
        else:
            population = next_generation

        # Reset world state
        supervisor.reset_world()

        # For each individual in the generation
        for individual in range(0, len(population)):

            # Create empty velocity message
            twist_msg = robot.create_zeroed_twist()

            # Set network weights to current inidividual
            neural_network.change_weights(population[individual][0], population[individual][1])

            # rospy.loginfo("Running gen {} individual {}".format(generation_count, individual))
            # print("Running gen {} individual {}".format(generation_count, individual))
            supervisor.update_progress(progress=[generation_count, individual])

            # Set ROS timer to ping every x minutes
            rospy.Timer(rospy.Duration(supervisor.return_evaluation_time()), supervisor.reset_timer, oneshot=True)

            # Run for 5 minutes or until critical failure
            while supervisor.check_end_condition() is False:
                # Get current laser data
                laser_data = robot.return_segmented_laser_data()

                # Feed forward through network and get next actions
                action = neural_network.feed_forward(laser_data)

                # Send actions to 'bot
                twist_msg.linear.x = action[0]
                twist_msg.angular.z = action[1]
                robot.publish_vel(twist_msg)

                bumper = robot.return_bumper_state()

                # Update the fitness of this action
                ga.fit_update(generation_fitness, individual, laser_data, action, bumper)
                # rospy.loginfo("Current fit: {}".format(generation_fitness[individual]))

                # node_rate.sleep()

            supervisor.reset_end_condition()
            # This individual is done, now reset for the next one
            supervisor.reset_world()

        # Sort the population and fitness structures in descending order i.e best
        # fitness first
        population, generation_fitness = ga.sort_by_fitness(population, generation_fitness)
        print(generation_fitness)
        rospy.loginfo("Best fit: {}".format(generation_fitness[0]))

        # Create the next generation by applying selection, crossover etc
        next_generation = ga.create_new_generation(population, generation_fitness)

        # Introduce some random mutations
        for individual in next_generation:
            ga.mutate(individual)

        # Reset fitness array
        generation_fitness.fill(0)
        generation_count += 1

    # Save best individuals, defaults to top 5 unless overridden
    supervisor.save_top_individuals(next_generation)
    rospy.loginfo("Evolution complete, top individuals saved")

if __name__ == "__main__":
    main()
    # test()
