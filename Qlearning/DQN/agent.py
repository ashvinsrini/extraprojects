from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
from config import *
import random
import tensorflow as tf
import numpy as np
class DQNAgent:
    def __init__(self):
        #Main model for prediction
        self.model = self.create_model()
        self.losses = []
        # Model for target predictions updated after different time steps
        self.target_model = self.create_model()
        #pdb.set_trace()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen = replay_memory_size)
        
        #self.tensorboard = ModifiedTensorBoard(log_dir = "logs/{}_{}".format(model_name, int(time.time())))
        self.target_update_counter = 0 ######### counter used to monitor the target model update


    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape= (2,), activation = "relu"))
        model.add(Dense(48, activation = "relu"))
        model.add(Dense(3, activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = 1e-3), metrics = ['accuracy'])
        return model
        
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def get_qs(self, state):
        action = self.model.predict(np.array(state.reshape(-1, *state.shape)))[0]
        #pdb.set_trace()
        return action
    
    def train(self, terminal_state, episode_no, success_epi_reward = None):
        #pdb.set_trace()
        #print(len(self.replay_memory))
        if len(self.replay_memory) < min_replay_mem_size:   
            return 
        
        minibatch = random.sample(self.replay_memory, minibatch_size)
        current_states = np.array([transition[0] for transition in minibatch])

        current_qs_list = self.model.predict(current_states)
        next_current_states = np.array([transition[3] for transition in minibatch])

        future_qs_list = self.target_model.predict(next_current_states)
        #pdb.set_trace()
        X = [] # input state parameters (pos, vel)
        y = [] # Q values 
        for index, (current_state, action, reward, next_current_state, done) in enumerate(minibatch):
            if not done: # or env.goal_position<0.5 or check done flag until True
                max_q = np.max(future_qs_list[index])
                new_q = reward + gamma*max_q
                
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            X.append(current_state)
            y.append(current_qs)
            #pdb.set_trace()
            
            
        callbacks = self.model.fit(np.array(X), np.array(y),epochs=1, verbose = 0) #[self.tensorboard] 
        self.losses.append(callbacks.history["loss"][0])
        
        if (success_epi_reward>-200) and (episode_no>=900):
            self.model.save('/home/ash/Project_1b/logs/{}_{}.model'.format(success_epi_reward, episode_no))
            print('model saved')
        
        '''
        if terminal_state:
            #pdb.set_trace()
            callbacks = self.model.fit(np.array(X), np.array(y), verbose = 0) #[self.tensorboard] 
            #pdb.set_trace()
            self.losses.append(callbacks.history["loss"][0])
            #pdb.set_trace()
            if episode_no%50==0:
                #pdb.set_trace()
                self.initial_loss = self.losses[-1]
                self.model.save('/home/ash/Project_1b/logs/{}.model'.format(self.initial_loss))
                #print('model saved')
            
        else:
            pass
        
        if terminal_state: 
            self.target_update_counter += 1
        if self.target_update_counter > update_tgt:
            print('updating target')
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        '''
        pass