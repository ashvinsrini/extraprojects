################ load a keras model #############
import keras
import gym
import numpy as np
from tqdm import tqdm
import argparse
import os
parser = argparse.ArgumentParser(description='code inputs information')
parser.add_argument('--p', dest='p', type=str, help='q table path')
parser.add_argument('--e', dest='e', type=str, help='number of episodes')
parser.add_argument('--r', dest='r', type=str, help='render is True or False')

args = parser.parse_args()
pth = args.p
episodes = eval(args.e)
render = args.r
env = gym.make("MountainCar-v0")
descrete_size = 30
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

done = False
DISCRETE_OS_SIZE = [descrete_size] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
render = True
q_table = np.load(os.path.join(pth))
for i in tqdm(range(episodes)):
    done = False
    current_state = env.reset()
    while not done:
        #q_values = model.predict(np.array(current_state.reshape(-1, *current_state.shape)))
        current_descrete_state = get_discrete_state(current_state)
        #action = np.argmax(q_values)
        action = np.argmax(q_table[current_descrete_state])
        
        new_state, reward, done, _ = env.step(action)
        current_state = new_state
        if render:
            env.render()
            pass
        else:
            pass
            
        #print(new_state[0])
    if new_state[0] > 0.5:
            print('reached goal')
    else:
            print('fail')