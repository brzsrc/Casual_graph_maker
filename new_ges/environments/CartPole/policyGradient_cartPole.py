# # REINFORCE on CartPole-v0
# 
# > In this post, We will take a hands-on-lab of Monte Carlo Policy Gradient (also known as REINFORCE) on openAI gym CartPole-v0 environment. This is the coding exercise from udacity Deep Reinforcement Learning Nanodegree.
# 
# - toc: true 
# - badges: true
# - comments: true
# - author: Chanseok Kang
# - categories: [Python, Reinforcement_Learning, PyTorch, Udacity]
# - image: images/CartPole-v0.gif

# ## REINFORCE
# ---
# In this notebook, you will implement REINFORCE agent on OpenAI Gym's CartPole-v0 environment. For summary, The **REINFORCE** algorithm ([Williams, 1992](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)) is a monte carlo variation of policy gradient algorithm in RL. The agent collects the trajectory of an episode from current policy. Usually, this policy depends on the policy parameter which denoted as $\theta$. Actually, REINFORCE is acronym for "**RE**ward **I**ncrement = **N**onnegative **F**actor * **O**ffset **R**einforcement * **C**haracteristic **E**ligibility"
# 
# ### Import the Necessary Packages
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 10)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
torch.manual_seed(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ### Instantiate the Environment and Agent
# 
# CartPole environment is very simple. It has discrete action space (2) and 4 dimensional state space. 
env = gym.make('CartPole-v1')
env.reset(seed=0)

print('observation space:', env.observation_space)
print('action space:', env.action_space)

# ### Define Policy
# Unlike value-based method, the output of policy-based method is the probability of each action. It can be represented as policy. So activation function of output layer will be softmax, not ReLU.
class Policy(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        # we just consider 1 dimensional probability of action
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)


# ### REINFORCE
def reinforce(policy, optimizer, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Collect trajectory
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        # Calculate total expected reward
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Recalculate the total reward applying discounted factor
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a,b in zip(discounts, rewards)])
        
        # Calculate the loss 
        policy_loss = []
        for log_prob in saved_log_probs:
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * R)
        # After that, we concatenate whole policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()
        
        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if e % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e - 100, np.mean(scores_deque)))
            break
    return scores

#Run the training process
def training_agent(policy):
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    scores = reinforce(policy, optimizer, n_episodes=2000)

# run game
'''
action sapce:
1:Push cart to the right
2:Push cart to the left

Observation space:
0: Cart Position
1: Cart Velocity
2: Pole Angle
3: Pole Angular Velocity
'''
def run_game(policy, env_name):
    list_before = []
    list_after = []
    env = gym.make(env_name)
    state = env.reset()
    done = False
    total_rewards = 0
    for t in range(1000):
        action, _ = policy.act(state)
        next_state, reward, done, _ = env.step(action)
        if(action == 0):
            action = 2
        list_before.append(state.tolist() + [total_rewards, action])
        total_rewards += reward
        list_after.append(next_state.tolist() + [total_rewards, action])
        state = next_state
        if done:
            break
    env.close()
    return np.array(list_before), np.array(list_after)

# dtype(data): np.array
def list_preprocess(data):
    dir = {i: data[data[:, -1] == i, :-1] for i in np.unique(data[:, -1])}
    return dir

def get_game_CPDAG(size):
    list_before, list_after = run_game(policy, 'CartPole-v0')
    dict_before = list_preprocess(list_before)
    dict_after = list_preprocess(list_after)

    dict_final = {}
    for action, data_before in dict_before.items():
        data_after = dict_after.get(action)
        data = np.concatenate((data_before, data_after), axis=0) 
        dict_final[action] = data

    return fit_bic(dict_final, size=size, phases=['forward', 'backward'], debug=0)


import sys
sys.path.append("../..")
from ges_algorithm import fit_bic
from DAG_maker import combine_CPDAG


size = 5
CPDAG_combiner = combine_CPDAG(size)
CPDAG_combiner2 = combine_CPDAG(size)

policy = Policy().to(device)
training_agent(policy)

cnt = 0 
while cnt < 100:
    print(cnt)
    CPDAG, action_matrix, score = get_game_CPDAG(size)
    if not all(action_matrix[CPDAG > 0] > 0):
        print('------------awwwwwwww------------')
        print(CPDAG)
        print(action_matrix)
        continue
    cnt+=1
    CPDAG_combiner.add_CPDAG(CPDAG, action_matrix, score)
    CPDAG_combiner2.add_CPDAG(CPDAG, action_matrix, score, include_undirected=False)

CPDAG_combiner.combine()
CPDAG_combiner2.combine()

graph1,action1,score1 = CPDAG_combiner.combine()    
graph2,action2,score2 = CPDAG_combiner2.combine()    

print("Final", np.all(graph1==graph2))
print(graph1,action1,score1)
print("---------------------------")
print(graph2,action2,score2)
