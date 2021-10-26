
import numpy as np
import gym
import pdb
import time
from tqdm import tqdm
from agent import DQNAgent
from tensorflow.python.keras import backend as K
import pickle
from config import *
import tensorflow as tf
# adjust values to your needs
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} ) 
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)


if __name__=="__main__":
    ep_rewards = []
    all_mets = {'epsilon':[], 'episode_rewards':[], 'loss':[], 'max_position':[]}
    #random.seed(10)
    #np.random.seed(10)
    #tf.random.set_seed(10)
    env = gym.make("MountainCar-v0")
    agent = DQNAgent()
    t = 1
    epsilon_list = []
    max_pos = []
    for episode in tqdm(range(1, episodes+1)):
        #agent.tensorboard.step = episode
        episode_reward = 0
        step_per_epi = 1
        current_state = env.reset()
        done = False
        pos = []
        epsilon = max(epsilon_min, epsilon)
        epsilon_list.append(epsilon)
        print('epsilon',epsilon)
        while not done: 

            pr = np.random.rand(1)
            if pr >= epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)
            pos.append(new_state[0])
            if new_state[0]>=0.5:
                reward+=10
                #print('reached goal position')

            episode_reward+= reward
            #pdb.set_trace()

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, episode_no = episode, success_epi_reward = reward)
            current_state = new_state
            step_per_epi+=1
            #print(done)
        print('updating target after every episode')
        agent.target_model.set_weights(agent.model.get_weights())
        max_pos.append(np.max(np.array(pos)))
        if new_state[0]<0.5:
            print('failed! goal not reached, max position:{}'.format(np.max(np.array(pos))))
        else:
            print('Success! goal reached at episode {}, max position:{}'.format(episode, np.max(np.array(pos))))
        #pdb.set_trace()
        ep_rewards.append(episode_reward)
        print(episode_reward)

        '''
        if epsilon > min_epsilon: 
            epsilon*=epsilon_decay
            epsilon = np.max((min_epsilon, epsilon))
        '''
        epsilon -= epsilon_decay
        t+=1
    all_mets['episode_rewards'] = ep_rewards
    all_mets['epsilon'] = epsilon_list
    all_mets['loss'] = agent.losses
    all_mets['max_position'] = max_pos
    with open('/home/ash/Project_1b/logs/all_mets.pickle', 'wb') as handle:
        pickle.dump(all_mets, handle, protocol=pickle.HIGHEST_PROTOCOL)