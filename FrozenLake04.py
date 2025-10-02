#This code is copied from url https://reinforcement-learning4.fun/2019/06/16/gym-tutorial-frozen-lake/
import numpy as np
import gymnasium
import random
import time
#from IPython.display import clear_output

original_map = ["SFFF", "FHFH", "FFFH", "HFFG"]
# custom_map = [
#     'SFFHFF',
#     'HFHFFF',
#     'HFFFHF',
#     'HFHFFH',
#     'HFFFFG'
# ]

custom_map = [
    'FFFFFF',
    'FFSFFF',
    'FFFHFF',
    'HFFFHF',
    'HFFFHF',
    'FFFFFG'

]
#env = gym.make('FrozenLake-v1', desc=custom_map,is_slippery=False)#This will set environment as custom_map defined above as 5X5 Grid
env = gymnasium.make("FrozenLake-v1", is_slippery=False,render_mode="human")

actions = {
    'Left': 0,
    'Down': 1,
    'Right': 2,
    'Up': 3
}
action_space_size = env.action_space.n#original code
#action_space_size = len(actions)#Added by mubbashir
action_space = env.action_space
#action_space = actions
state_space_size = env.observation_space.n
print("action space =",env.action_space.n)
print("state space size =",env.observation_space.n)

num_episodes = 1000
max_steps_per_episode = 100

# <editor-fold desc="Description">
for episode in range(2):

    # initialize new episode params
    env.reset()
    done = False
    print("*****EPISODE ", episode + 1, "*****\n\n\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):
        #clear_output(wait=True)
        random_action = env.action_space.sample()
        #print("random action = ", random_action)
        #new_state, reward, done, info = env.step(random_action)#Original code
        new_state, reward, terminated, truncated, info = env.step(random_action)  # Modified code to include own reward
        #new_state = findState(new_state)
        if(terminated or truncated):
            reward=new_state*2;
            done=True;
        else :
            reward=2
      #  prob, new_state, reward, done = env.P[state][random_action][0]#Testing code by mubbashir
        # new_state, reward, done, info = env.step(action)
        print("new/next state,", "reward,", "terminated,", "Probability,", "Action picked,","step number")
        print(new_state, reward, terminated, info, random_action,step)
        env.render();# To print the grid on the console
        time.sleep(0.3)


        if done:
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
        # Set new state
        state=new_state;#Original code
        #state=findState(state)



time.sleep(2)
env.close()
#print(GridPos[0])



# </editor-fold>