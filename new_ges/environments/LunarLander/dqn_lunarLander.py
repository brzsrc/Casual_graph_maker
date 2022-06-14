# # Deep Q-Network (DQN) on LunarLander-v2
# 
# > In this post, We will take a hands-on-lab of Simple Deep Q-Network (DQN) on openAI LunarLander-v2 environment. This is the coding exercise from udacity Deep Reinforcement Learning Nanodegree.
# 
# - toc: true 
# - badges: true
# - comments: true
# - author: Chanseok Kang
# - categories: [Python, Reinforcement_Learning, PyTorch, Udacity]
# - image: images/LunarLander-v2.gif

# ## Deep Q-Network (DQN)
# ---
# In this notebook, you will implement a DQN agent with OpenAI Gym's LunarLander-v2 environment.
# 
# ### Import the Necessary Packages
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import deque, namedtuple


# ### Instantiate the Environment and Agent
# 
# Initialize the environment.
env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

# ### Define Neural Network Architecture.
# 
# Since `LunarLander-v2` environment is sort of simple envs, we don't need complicated architecture. We just need non-linear function approximator that maps from state to action.
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

# ### Define some hyperparameter
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ### Define Agent 
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# ### Define Replay Buffer
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# ### Training Process
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

# run the game        
'''
Action Space: size 4
action index starts from 1 
do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

Observation Space:state size 8 
the coordinates of the lander in x & y, 
its linear velocities in x & y, 
its angle, its angular velocity,
two booleans that represent whether each leg is in contact with the ground or not.
'''
def run_game(agent, env_name):
    list_before = []
    list_after = []
    env = gym.make(env_name)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    state = env.reset()
    done = False
    total_rewards = 0
    while not done:     
        action = agent.act(state)
        action_ = action
        if(action == 0):
            action_ = 5
        state_list = state.tolist()
        # landed = 0
        # if(state_list[6] and state_list[7]):
        #     landed = 1
        list_before.append(state_list[0:6] + [total_rewards, action_])
        state, reward, done, _ = env.step(action)  
        state_list = state.tolist() 
        total_rewards += reward
        # landed = 0
        # if(state_list[6] and state_list[7]):
        #     landed = 1
        # print(state)
        list_after.append(state_list[0:6] + [total_rewards, action_])
    # print("finsihed")     
    env.close()
    return np.array(list_before), np.array(list_after)

# dtype(data): np.array
def list_preprocess(data):
    dir = {i: data[data[:, -1] == i, :-1] for i in np.unique(data[:, -1])}
    return dir

def get_game_CPDAG(size):
    agent = Agent(state_size=8, action_size=4, seed=0)
    list_before, list_after = run_game(agent, 'LunarLander-v2')
    dict_before = list_preprocess(list_before)
    dict_after = list_preprocess(list_after)

    dict_final = {}
    for action, data_before in dict_before.items():
        data_after = dict_after.get(action)
        data = np.concatenate((data_before, data_after), axis=0) 
        dict_final[action] = data

    for action, data in dict_final.items():
        # print(data)
        trans_arr = data.T
        for i in range(trans_arr.shape[0]):
            if np.all(trans_arr[i] == trans_arr[i][0]):
                print("action:", action, 'Column: ', i)
    # print(dict_final)
    return fit_bic(dict_final, size=size, phases=['forward', 'backward'], debug=0)

import sys
sys.path.append("../..")
from ges_algorithm import fit_bic

from DAG_maker import combine_CPDAG

size = 8
CPDAG_combiner = combine_CPDAG(size)
CPDAG_combiner2 = combine_CPDAG(size)

cnt = 0 
while cnt < 100:
    # print(cnt)
    CPDAG, action_matrix, score = get_game_CPDAG(size)
    if not all(action_matrix[CPDAG > 0] > 0):
        print('------------awwwwwwww------------')
        print(CPDAG)
        print(action_matrix)
        continue
    cnt+=1
    CPDAG_combiner.add_CPDAG(CPDAG, action_matrix, score)
    CPDAG_combiner2.add_CPDAG(CPDAG, action_matrix, score, include_undirected=False)
    
graph1,action1,score1 = CPDAG_combiner.combine()    
# graph2,action2,score2 = CPDAG_combiner2.combine()    

# print("Final", np.all(graph1==graph2))
print(graph1,action1,score1)
# print("---------------------------")
# print(graph2,action2,score2)