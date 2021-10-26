from keras.models import sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import numpy as np

class Qlearning:
    def __init__(self, q_table, lr, gamma):
        self.q_table = q_table
        self.lr = lr
        self.gamma = gamma
        pass

    def max_q(self, new_discrete_state):
        self.max_future_q = np.max(self.q_table[new_discrete_state])

    def update_new_q(self, discrete_state, action, reward, done, new_state, final_pos):
         if not done:
             current_q = self.q_table[discrete_state + (action,)]
             new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * self.max_future_q)
             self.q_table[discrete_state + (action,)] = new_q
         elif new_state[0]>final_pos:
             self.q_table[discrete_state + (action,)] = 0
         #self.new_state = new_state



class SARSA:
    def __init__(self, q_table, lr, gamma):
        self.q_table = q_table
        self.lr = lr
        self.gamma = gamma
        pass

    def max_q(self, new_discrete_state, pr, env, epsilon):
        if pr >= epsilon:
            self.max_future_q = np.max(self.q_table[new_discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
            self.max_future_q = self.q_table[new_discrete_state + (action,)]
        #print('alternate max q chosen', pr, epsilon)

    def update_new_q(self, discrete_state, action, reward, done, new_state, final_pos):
        if not done:
            current_q = self.q_table[discrete_state + (action,)]
            new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * self.max_future_q)
            self.q_table[discrete_state + (action,)] = new_q
        elif new_state[0] > final_pos:
            self.q_table[discrete_state + (action,)] = 0
        # self.new_state = new_state


class Expected_SARSA:
    def __init__(self, q_table, lr, gamma):
        self.q_table = q_table
        self.lr = lr
        self.gamma = gamma
        pass

    def max_q(self, new_discrete_state, pr, env, epsilon):
        all_actions = [0,1,2]
        action = np.argmax(self.q_table[new_discrete_state])
        other_actions = np.array([a for a in all_actions if a != action])

        self.max_future_q = 0.8*np.max(self.q_table[new_discrete_state]) + \
                            0.1*self.q_table[new_discrete_state + (other_actions[0],)] + \
                            0.1*self.q_table[new_discrete_state + (other_actions[1],)]


    def update_new_q(self, discrete_state, action, reward, done, new_state, final_pos):
        if not done:
            current_q = self.q_table[discrete_state + (action,)]
            new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * self.max_future_q)
            self.q_table[discrete_state + (action,)] = new_q
        elif new_state[0] > final_pos:
            self.q_table[discrete_state + (action,)] = 0
        # self.new_state = new_state