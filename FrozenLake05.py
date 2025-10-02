#Below code is copied from https://www.kaggle.com/sarjit07/reinforcement-learning-using-q-table-frozenlake

import gym
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
# custom_map = [
#     'SFFHF',
#     'HFHFF',
#     'HFFFG',
#     'HHHFF',
#     'HFFFH'
# ]
#env = gym.make('FrozenLake-v1',is_slippery=False, desc=custom_map)#This will set environment as custom_map defined above as 5X5 Grid
#env = gym.make('FrozenLake-v1', desc=custom_map)
#PyTorch is a Python package that provides two high-level features:
#Tensor computation (like NumPy) with strong GPU acceleration
#Deep neural networks built on a tape-based autograd system
#env = gym.make("FrozenLake-v1", is_slippery=False)#To make frozen lake problem deterministic, otherwise agent can fell into a hole
env = gym.make("FrozenLake-v1",is_slippery=False)
# Total number of States and Actions
#number_of_states = env.observation_space.n
#number_of_actions = env.action_space.n
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n
print( "States = ", number_of_states)
print( "Actions = ", number_of_actions)

num_episodes = 500
steps_total = []
rewards_total = []
egreedy_total = []
# if learning_rate == 0:
#      Pick value of new Q(s,a) based on past experience
# elif learning_rate == 1:
#      Pick value of new Q(s,a) based on current situtation

# Value of learning_rate(alpha) varies from [0 - 1]
# Discount rate accounts for the Reward the agent receive on an action

# if discount_rate == 0:
#     only current reward accounted
# elif discount_rate == 1:
#     future rewards also accounted
# PARAMS

# Discount on reward
gamma = 0.95

# Factor to balance the ratio of action taken based on past experience to current situtation
learning_rate = 0.9
# exploit vs explore to find action
# Start with 70% random actions to explore the environment
# And with time, using decay to shift to more optimal actions learned from experience

egreedy = 0.7
egreedy_final = 0.1
egreedy_decay = 0.999
Q = torch.zeros([number_of_states, number_of_actions])
for i_episode in range(num_episodes):

    # resets the environment
    state = env.reset()
    state=state[0]
    step = 0

    while True:

        step += 1

        random_for_egreedy = torch.rand(1)[0]#To generate a random number between 0-1
        #print("Random for greedy = ", random_for_egreedy);
        if random_for_egreedy > egreedy:    # Below code chooses action which has the highest Q value. i.e. exploitation
            random_values = Q[state] + torch.rand(1, number_of_actions) / 1000#generate 4small random numbers
            #print("random_values = ",random_values)
            action = torch.max(random_values, 1)[1][0]#action having the highest Q-value
            action = action.item()
        else: # Below code chooses action at random i.e exploration
            action = env.action_space.sample()

        if egreedy > egreedy_final:
            egreedy *= egreedy_decay

        new_state, reward, terminated, truncated, info= env.step(action)#info gives probability of the state
       # print(new_state, reward, done, info)
        # Filling the Q Table
        #print("new_state = ",new_state)
        #Q[state, action] = reward + gamma * torch.max(Q[new_state])
        Q[state, action] =Q[state, action] + \
        learning_rate * (reward + gamma * torch.max(Q[new_state]) - Q[state, action])
        # Setting new state for next action
        state = new_state

        # env.render()
        # time.sleep(0.4)

        if terminated or truncated:
            steps_total.append(step)
            rewards_total.append(reward)
            egreedy_total.append(egreedy)
            #print('Episode: {} Reward: {} Steps Taken: {}'.format(i_episode, reward, step))# Added by mubbashir
            #break
            if i_episode % 10 == 0:
                  print('Episode: {} Reward: {} Steps Taken: {}'.format(i_episode, reward, step))# Original code
            break

print(Q)

print("Percent of episodes finished successfully: {0}".format(sum(rewards_total) / num_episodes))
print("Percent of episodes finished successfully (last 100 episodes): {0}".format(sum(rewards_total[-100:]) / 100))
print("Average number of steps: %.2f" % (sum(steps_total) / num_episodes))
print("Average number of steps (last 100 episodes): %.2f" % (sum(steps_total[-100:]) / 100))
# # Below graph plots number of episodes vs rewards; X-axis contain episodes, Y-axis rewards, max value of reward=1
plt.figure(figsize=(10,5))
plt.title("Rewards/Episodes")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
#plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green', width=5)
plt.plot(torch.arange(len(rewards_total)), rewards_total, color='green')
plt.show()
# Below graph plots number of episodes vs steps; X-axis contain number of episodes, Y-axis contain Episode steps
#Here, we can see that the steps taken in an episode is large because we have high random action in the starting and later on
#...as we learn from experience, we start to take more informed steps , hence less noise
plt.figure(figsize=(10,5))
plt.title("Episodes Steps / Episodes length")
plt.xlabel("Episodes")
plt.ylabel("Episode Steps")
#plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red', width=5)
plt.plot(torch.arange(len(steps_total)), steps_total)
plt.show()

plt.figure(figsize=(10,5))
plt.title("Egreedy value/Episodes")
plt.xlabel("Episodes")
plt.ylabel("Egreedy value")
#plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha=0.6, color='blue', width=5)
plt.plot(torch.arange(len(egreedy_total)), egreedy_total)
plt.show()
def extractPolicy(Q,env) :
    #Mubbahsir start from state in reset, check start should be a number or np.asscalar
    state = env.reset()
    state = state[0]
    print("state = ",state," Q = ",Q[state])
    done = False
    sequence = []

    while not done:
        # Choose the action with the highest value in the current state
        print("Q[state]) = ",Q[state])
        if torch.max(Q[state]) > 0:
            action = torch.argmax(Q[state])
            action =action.item()
            print("picked action = ",action)
            #print(" state  = ", state, "and Q[State] > 0"," action = ",action,"value at state and action = ",Q[np.array(state)][action])

        # If there's no best action (only zeros), take a random one
        else:
            #print(" state  = ", state, "and Q[State] <= 0")
            action = env.action_space.sample()
            print("random action = ",action)
        new_state, reward, terminated, truncated, info = env.step(action)
        if reward==1:
            done = True;
        state=new_state
        # Add the action to the sequence
        sequence.append(action)

    return sequence
########################33
def applyPolicy(policy,env):
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name="4x4", render_mode="human")
    # env=gym.make("ALE/Adventure-v5")# render_mode="human")
    env.reset()
    env.render()
    for a in policy:
        # new_state, reward, done, info = env.step(actions[a])
        new_state, reward, terminated, truncated, info = env.step(policy[a])
        print("new_state = ", new_state)
        env.render()
        #print("Reward: {:.2f}".format(reward))
        #print(info, new_state)
        #print(terminated, truncated)
        if terminated:
            break
    time.sleep(6)

policy=extractPolicy(Q,env)
print("Policy = ", policy)
print("applying policy = ",policy)
applyPolicy(policy,env)