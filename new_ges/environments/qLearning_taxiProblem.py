# # Q* Learning with OpenAI Taxi-v2 

# ## Step 0: Import the dependencies 
# First, we need to import the libraries <b>that we'll need to create our agent.</b></br>
# We use 3 libraries:
# - `Numpy` for our Qtable
# - `OpenAI Gym` for our Taxi Environment
# - `Random` to generate random numbers

import numpy as np
import gym
import random

# ## Step 1: Create the environment 
# - Here we'll create the Taxi environment. 
# - OpenAI Gym is a library <b> composed of many environments that we can use to train our agents.</b>

env = gym.make("Taxi-v3")
# env.reset()
# env.render()

# ## Step 2: Create the Q-table and initialize it 
# - Now, we'll create our Q-table, to know how much rows (states) and columns (actions) we need, we need to calculate the action_size and the state_size
# - OpenAI Gym provides us a way to do that: `env.action_space.n` and `env.observation_space.n`

action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size ", state_size)

qtable = np.zeros((state_size, action_size))
# print(qtable)

# ## Step 3: Create the hyperparameters 
# Here, we'll specify the hyperparameters.

total_episodes = 5000        # Total episodes
total_test_episodes = 100   # Total test episodes
max_steps = 99                # Max steps per episode

learning_rate = 0.7           # Learning rate
gamma = 0.618                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# ## Step 4: The Q learning algorithm 
# - Now we implement the Q learning algorithm:

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    # print(state)
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0,1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
        
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * 
                                    np.max(qtable[new_state, :]) - qtable[state, action])
                
        # Our new state is state
        state = new_state
        
        # If done : finish episode
        if done == True: 
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 

# ## Step 5: Use our Q-table to play Taxi ! 
# - After 50 000 episodes, our Q-table can be used as a "cheatsheet" to play Taxi.
# - By running this cell you can see our agent playing Taxi.

#state = taxi_row, taxi_col, pass_idx, dest_idx
rewards = []
locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

def run_game():
    env.reset()
    list_before = []
    list_after = []
    for episode in range(total_test_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        # print("****************************************************")
        # print("EPISODE ", episode)

        for step in range(max_steps):
            action = np.argmax(qtable[state,:])
            new_state, reward, done, info = env.step(action) 

            # type(env.decode(state): list_reverseiterator
            taxi_row, taxi_col, pass_idx, dest_idx = env.decode(state)
            new_taxi_row, new_taxi_col, new_pass_idx, new_dest_idx = env.decode(new_state)

            if action != 4 and action != 5:
                action = 6
            list_before.append([taxi_row, taxi_col, pass_idx, dest_idx, total_rewards, action])
            total_rewards += reward
            list_after.append([new_taxi_row, new_taxi_col, new_pass_idx, new_dest_idx, total_rewards, action])

            state = new_state
            if done:
                break

            step+=1
    env.close()
    return np.array(list_before), np.array(list_after)

# n = 5, 25 taxi positions
# taxi_locats: a list of array
def encode_location(taxi_locats, n):
    locat_arr = np.arange(1, n * n + 1).reshape(n,n)
    locat_list = [locat_arr[i, j] for i, j in taxi_locats]
    return np.array(locat_list)

# dtype(data): np.array
def list_preprocess(raw_data):
    split_data = np.split(raw_data, [2], axis=1) 
    data_locat = encode_location(split_data[0].tolist(), 5)
    data = np.hstack((data_locat.reshape(data_locat.shape[0], 1), split_data[1]))
    dir = {i: data[data[:, -1] == i, :-1] for i in np.unique(data[:, -1])}
    return dir

import sys
sys.path.append("..")
from ges_algorithm import fit_bic
from utils import is_dag

def get_game_CPDAG(size):
    list_before, list_after = run_game()
    dict_before = list_preprocess(list_before)
    dict_after = list_preprocess(list_after)

    dict_final = {}
    for action, data_before in dict_before.items():
        data_after = dict_after.get(action)
        data = np.concatenate((data_before, data_after), axis=0) 
        dict_final[action] = data

    return fit_bic(dict_final, size=size, phases=['forward', 'backward'], debug=0)


from DAG_maker import combine_CPDAG

size = 4
CPDAG_combiner = combine_CPDAG(size)
CPDAG_combiner2 = combine_CPDAG(size)

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
    
graph1,action1,score1 = CPDAG_combiner.combine()    
graph2,action2,score2 = CPDAG_combiner2.combine()    

print("Final", np.all(graph1==graph2))
print(graph1,action1,score1)
print("---------------------------")
print(graph1,action1,score1)