################ load a keras model #############
import keras
import gym
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='code inputs information')
parser.add_argument('--m', dest='m', type=str, help='model path')
parser.add_argument('--e', dest='e', type=str, help='number of episodes')
parser.add_argument('--r', dest='r', type=str, help='render is True or False')

args = parser.parse_args()

model_pth = args.m
episodes = eval(args.e)
render = args.r
model = keras.models.load_model(model_pth)
env = gym.make("MountainCar-v0")

for i in tqdm(range(episodes)):
    done = False
    current_state = env.reset()
    while not done:
        q_values = model.predict(np.array(current_state.reshape(-1, *current_state.shape)))
        action = np.argmax(q_values)
        new_state, reward, done, _ = env.step(action)
        current_state = new_state
        if render:
            env.render()
        else:
            pass
            
        #print(new_state[0])
    if new_state[0] > 0.5:
            print('reached goal')
    else:
            print('fail')