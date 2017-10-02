import gym
import numpy as np

env = gym.make('FrozenLake-v0')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
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
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,reward,d,_ = env.step(action)
        #Update Q-Table with new knowledge
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
