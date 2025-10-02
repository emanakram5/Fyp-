# In this code frozen lake problem is made deterministic by setting is_slippery=False, in gym.make() method
import numpy as np
import gymnasium
import random
import time
#from IPython.display import clear_output
custom_map = [
    'SFFHFF',
    'FFHFFF',
    'HFFFHF',
    'FHFFFG',
    'HFFFFH'
]
#Original frozen lake environment
#env = gym.make("FrozenLake-v1", is_slippery=False)#To make frozen lake problem deterministic, otherwise agent can fell into a hole
#Frozen lake with custom map
env = gymnasium.make('FrozenLake-v1', desc=custom_map,render_mode="human")#This will set environment as custom_map defined above as 5X5 Grid
actions = {
    'Left': 0,
    'Down': 1,
    'Right': 2,
    'Up': 3
}
#action_space_size = len(actions)#Added by mubbashir
action_space = env.action_space
action_space_size = env.action_space.n
#action_space = actions
state_space_size = env.observation_space.n
print("action space =",action_space)
q_table = np.zeros((state_space_size, action_space_size))
#print(q_table)
num_episodes = 10
max_steps_per_episode = 100


# Watch our agent play Frozen Lake by playing the best action
# from each state according to the Q-table

# <editor-fold desc="Description">
for episode in range(3):
    # initialize new episode params
    state = env.reset()

    done = False
    print("*****EPISODE ", episode + 1, "*****\n\n\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):
        # Show current state of environment on screen
        # Choose action with highest Q-value for current state
        # Take new action

        #clear_output(wait=True)
        random_action = env.action_space.sample()
        print("random action = ", random_action)
        OBS, reward, terminated, truncated, info = env.step(random_action)#Original code
      #  prob, new_state, reward, done = env.P[state][random_action][0]#Testing code by mubbashir
        # new_state, reward, done, info = env.step(action)
        print(OBS, reward, terminated, truncated, info, random_action)
        #print(env.env.P[0][1])#output is in the formP[s|a] = P[s’], s’, r, done, ---> probability, state, reward and done= flase or true
        #env.env.P[0][1] means from state 0 and action 1 what is the probabilty of next state, reward, done
        # [(1.0, 4, 0.0, False)] 1.0 is probability, 4 is state, 0.0 is reward and false= done

        env.render();# To print the grid on the console
        time.sleep(0.3)
        #action = np.argmax(q_table[state, :])

        if terminated or truncated:
            #clear_output(wait=True)
            env.render()
            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
                #clear_output(wait=True)
            break
#print(q_table)
env.close()
# </editor-fold>