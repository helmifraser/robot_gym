# robot-gym
Gazebo environment for testing reinforcement learning and evolutionary robotics.

Demonstrates a neural network controlled robot evolving to navigate collision free in a maze environment.

This was a small project during my MSc in 2017, the slides for the presentation I gave related to this repo can be found [here](https://docs.google.com/presentation/d/1kyWxEuvd2_vbKHwcCCBYBkj77z_3xrTFpY_o0ZVGCIg/edit?usp=sharing).

**HARD CODE 'kobuki.launch.xml' in 'turtlebot_gazebo' to point to /opt/ros/kinetic/share/turtlebot_description/robots/kobuki_hexagons_hokuyo.urdf.xacro**


# Results

During evolution (left) and after (right)

<p float="left">
  <img src="/misc/during_evolution.gif" width="49%" />
  <img src="/misc/after_evolution.gif" width="49%" /> 
</p>

Opposite direction (left) and escaping a trap (right)

<p float="left">
  <img src="/misc/opposite.gif" width="49%" />
  <img src="/misc/escape.gif" width="49%" /> 
</p>

Failure cases: open space and unfamiliar environments 

<p float="left">
  <img src="/misc/open.gif" width="49%" />
  <img src="/misc/fail.gif" width="49%" /> 
</p>

A lot can be improved, and it's clearly overfit to the "maze" environment, but hey, it's a proof of concept. Of course, you could use a classic controller or optimise the weights another way.

# Disclaimer

This repo and all contained was for a short project/assignment and was never intended to be publicly released (hence the mess, lack of documentation and general unusability), so everything here is provided as-is. This is unlikely to be worked on in the forseeable future.
