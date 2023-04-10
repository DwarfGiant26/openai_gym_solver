from tools.remote_display import *
import numpy as np

class QLearning():
    '''
        This is a Q learning model class. 
        
        Fields
            state_bins : 2d numpy array with each row being list of boundaries for each state in order
                e.g. :
                    state_bins = [
                        np.linspace(-4.8, 4.8, 20),
                        np.linspace(-0.5, 0.5, 20),
                        np.linspace(-0.42, 0.42, 20),
                        np.linspace(-0.5, 0.5, 20)
                    ]
            q_table : n dimensional numpy array. The first n-1 dimension are the states. The last dimension is action
            train_frames_episodes : list of episodes. Each episode is a list of frames. Each frame is a 3D array.
            test_frames : list of frames from testing

        
        Methods:
            It contains method to train and test the model
    '''
    def __init__(self, state_bins, n_actions):
        
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.q_table = self.create_q_table()
        self.train_frames_episodes = []
        self.test_frames = []
        
    def create_q_table(self):
        '''
            Output: n dimensional numpy array. The first n-1 dimension are the states. The last dimension is action
        '''
        # +1 because state bin is only the boundary. The number of spaces is boundary + 1
        dimension = [state_bin.shape[0]+1 for state_bin in self.state_bins]
        dimension.append(self.n_actions)
        
        return np.zeros(dimension)
        
        
    def discretize_state(state, state_bins):
        '''
            Discretize continuous state space so that we can create q table with finite size
            Output: 
        '''
        return tuple(np.digitize(s, bins) for s, bins in zip(state, state_bins))

    def choose_action(state, q_table, epsilon, n_actions):
        '''
            Choose an action using epsilon-greedy policy, given current state
            
            state: current state
            eplison: exploration rate
            
        '''
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        else:
            return np.argmax(q_table[state])

    # Update Q-table using Q-learning algorithm
    def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
        '''
            Update the table
        '''
        q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
    
    def train(self, env, n_episodes, max_steps, alpha, gamma, epsilon, epsilon_min, epsilon_decay, real_time_video=False, keep_frames=True):
        '''
            Train Q-learning agent. The training will directly update the q learning model object fields such as q table, and frames.
            Input
                max_steps : maximum step in an episode
                alpha : learning rate
                gamma : discount factor
                epsilon : exploration rate
                epsilon_min : the minimum rate of epsilon that can no longer decay
                epsilon_decay : the rate in which epsilon decay in an episode
                real_time_video : boolean for whether to display video while training
                keep_frames : boolean for whether to keep the frames to be shown later on
            
            Output: -
        '''
        for episode in range(n_episodes):
            
            # prepare new game
            observation, info = env.reset()
            state = QLearning.discretize_state(observation, self.state_bins)
            
            # prepare list to capture frames
            frames = []

            for i in range(max_steps):
                
                # choose action and update state
                action = QLearning.choose_action(state, self.q_table, epsilon, self.n_actions)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # render frame
                frame = env.render()
                # show frame in real time if real time video is true
                if real_time_video:
                    show_frame(frame)
                # keep frames
                if keep_frames:
                    frames.append(frame)
                
                # terminate if game end
                if terminated or truncated:
                    break
                    
                # update state, and q table
                next_state = QLearning.discretize_state(next_state, self.state_bins)
                QLearning.update_q_table(self.q_table, state, action, reward, next_state, alpha, gamma)
                state = next_state
            
            # Decay exploration rate
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            # update frame episodes
            if keep_frames:
                self.train_frames_episodes.append(frames)