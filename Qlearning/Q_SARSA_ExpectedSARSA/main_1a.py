import gym
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pandas as pd
from qlearn_SARSA import Qlearning,SARSA, Expected_SARSA
env = gym.make("MountainCar-v0")
import random
import os
from tqdm import tqdm
descrete_sizes = [10, 20, 30, 40, 50, 60,70,80,90]
methods = ['Q','SARSA','ESARSA']
plt_counter = 1
for method in tqdm(methods):
    for descrete_size in descrete_sizes:
        random.seed(10)
        np.random.seed(10)
        lr = 0.1
        gamma = 0.9
        episodes = 10_000
        SHOW_EVERY = 500
        #method = 'Q'
        def get_discrete_state(state):
            discrete_state = (state - env.observation_space.low)/discrete_os_win_size
            return tuple(discrete_state.astype(np.int))

        done = False
        DISCRETE_OS_SIZE = [descrete_size] * len(env.observation_space.high)
        discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
        #epsilon = 0.5
        start_epsilon_decay = 1
        end_epsilon_decay = episodes//2
        #epsilon_decay_value = epsilon/(end_epsilon_decay - start_epsilon_decay)
        #print(discrete_os_win_size)

        q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
        if method == 'Q':
            qlearn = Qlearning(q_table, lr, gamma)
        elif method == 'SARSA':
            qlearn = SARSA(q_table, lr, gamma)
        else:
            qlearn = Expected_SARSA(q_table, lr, gamma)
        all_episodes_rewards = []
        dict_ep_rewards = {'ep' : [], 'mean' : [], 'min' : [], 'max' : []}

        for episode in range(episodes):
            epsilon, t = 0.05, 1
            ep_r = 0

            if episode%SHOW_EVERY == 0:
                #print(episode)
                render = True
                per_ep_rewards = []

            else:
                render = False

            discrete_state = get_discrete_state(env.reset())
            done = False
            step_per_epi = 0
            while not done:
                epsilon/=t
                pr = np.random.uniform(low =0.0, high = 1.0, size = None)
                if pr >= epsilon:
                    action = np.argmax(q_table[discrete_state])
                else:
                    action = np.random.randint(0, env.action_space.n)
                new_state, reward, done, _ = env.step(action)
                #print('reward is', reward)
                ep_r+=reward
                new_discrete_state = get_discrete_state(new_state)
                if (episode>500) and (episode%10==0):
                    #env.render()
                    pass
                if method == 'Q':
                    qlearn.max_q(new_discrete_state)
                else:
                    qlearn.max_q(new_discrete_state, pr, env, epsilon)
                qlearn.update_new_q(discrete_state, action, reward, done, new_state, final_pos = env.goal_position)
                if new_state[0] >= env.goal_position:
                    #print('reward:', reward)
                    #print('reached goal post on episode', episode)
                    pass
                discrete_state = new_discrete_state
                step_per_epi+=1
            #pdb.set_trace()
            per_ep_rewards.append(ep_r)
            all_episodes_rewards.append(ep_r)
            #print('episode',episode,'ep_r',ep_r)
            t+=1
            if len(per_ep_rewards) == SHOW_EVERY:
                mean_reward = np.mean(per_ep_rewards)
                dict_ep_rewards['ep'].append(episode)
                dict_ep_rewards['mean'].append(mean_reward)
                dict_ep_rewards['min'].append(np.min(per_ep_rewards))
                dict_ep_rewards['max'].append(np.max(per_ep_rewards))

                #print("Episode:{}, mean:{}, min:{}, max:{}".format(episode, mean_reward,np.min(per_ep_rewards),np.max(per_ep_rewards)))
                #pdb.set_trace()

            #if end_epsilon_decay >= episode >= start_epsilon_decay:
                #epsilon -= epsilon_decay_value
        env.close()


        if not os.path.exists('./results'):
            os.makedirs('./results')
        save_pth = os.path.join(os.getcwd(),'results')

        plt.figure(plt_counter)
        plt.plot(dict_ep_rewards['ep'], dict_ep_rewards['mean'], label = 'mean')
        plt.plot(dict_ep_rewards['ep'], dict_ep_rewards['min'], label = 'min')
        plt.plot(dict_ep_rewards['ep'], dict_ep_rewards['max'], label = 'max')
        plt.legend()
        plt.savefig(os.path.join(save_pth, 'averaged_500_episodes_{}_{}.png'.format(method, descrete_size)))
        plt.clf()

        #plt.show()


        '''
        plt.figure(2)
        plt.plot([_ for _ in range(len(all_episodes_rewards))], all_episodes_rewards)
        plt.savefig(os.path.join(save_pth, 'all_episodes_rewards_{}_{}states.png'.format(method,descrete_size)))
        plt.show()
        '''

        series = pd.Series(all_episodes_rewards).rolling(window = 1000).mean()
        plt.figure(plt_counter+1)
        plt.plot(series)
        plt.savefig(os.path.join(save_pth, 'all_episodes_rewards_rollingmean_{}_{}states.png'.format(method, descrete_size)))
        plt.clf()
        #plt.show()
        plt_counter+2
