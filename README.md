# robot-gym
Gazebo environment for testing reinforcement learning and evolutionary robotics

# Installation

***Most of these are now redundant since the OpenAI and Tensorflow dependencies were removed***

1. create a virtualenv for the whole thing (use virtualenvwrapper)
2. enable global site packages (toggleglobalsitepackages)
3. clone openai gym into it (https://github.com/openai/gym) and follow install
  instructions
4. pip install tensorflow
5. pip install h5hpy
6. pip install keras
5. clone gym-gazebo (https://github.com/erlerobot/gym-gazebo) and remember to
  pip install -e .
6. run turtlebot_setup.bash in gym_gazebo/envs/installation
7. pip install scikit-image
8. probably something else I've forgotten but deal with any errors one by one
