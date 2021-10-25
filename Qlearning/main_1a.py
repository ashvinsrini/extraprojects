import gym
import matplotlib.pyplot as plt
import numpy as np
import pdb
from DQN import Qlearning,SARSA
env = gym.make("MountainCar-v0")

lr = 0.1
gamma = 0.9
episodes = 2_000
SHOW_EVERY = 100
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

done = False
DISCRETE_OS_SIZE = [10] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
#epsilon = 0.5
start_epsilon_decay = 1
end_epsilon_decay = episodes//2
#epsilon_decay_value = epsilon/(end_epsilon_decay - start_epsilon_decay)
print(discrete_os_win_size)

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
qlearn = Qlearning(q_table, lr, gamma)
#qlearn = SARSA(q_table, lr, gamma)

dict_ep_rewards = {'ep' : [], 'mean' : [], 'min' : [], 'max' : []}

for episode in range(episodes):
    epsilon, t = 0.05, 1
    ep_r = 0

    if episode%SHOW_EVERY == 0:
        print(episode)
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
            env.render()
            pass

        qlearn.max_q(new_discrete_state)
        #qlearn.max_q(new_discrete_state, pr, env, epsilon)
        qlearn.update_new_q(discrete_state, action, reward, done, new_state, final_pos = env.goal_position)
        if new_state[0] >= env.goal_position:
            print('reward:', reward)
            print('reached goal post on episode', episode)

        discrete_state = new_discrete_state
        step_per_epi+=1
    #pdb.set_trace()
    per_ep_rewards.append(ep_r)
    print('ep_r',ep_r)
    t+=1
    if len(per_ep_rewards) == SHOW_EVERY:
        mean_reward = np.mean(per_ep_rewards)
        dict_ep_rewards['ep'].append(episode)
        dict_ep_rewards['mean'].append(mean_reward)
        dict_ep_rewards['min'].append(np.min(per_ep_rewards))
        dict_ep_rewards['max'].append(np.max(per_ep_rewards))
        print("Episode:{}, mean:{}, min:{}, max:{}".format(episode, mean_reward,np.min(per_ep_rewards),np.max(per_ep_rewards)))
        #pdb.set_trace()

    #if end_epsilon_decay >= episode >= start_epsilon_decay:
        #epsilon -= epsilon_decay_value
env.close()

plt.figure()
plt.plot(dict_ep_rewards['ep'], dict_ep_rewards['mean'], label = 'mean')
plt.plot(dict_ep_rewards['ep'], dict_ep_rewards['min'], label = 'min')
plt.plot(dict_ep_rewards['ep'], dict_ep_rewards['max'], label = 'max')
plt.legend()
plt.show()
