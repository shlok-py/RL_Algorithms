import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


# Implementation with Frozen Lake

env = gym.make('FrozenLake', is_slippery=True)

action_size = env.action_space.n
state_size = env.observation_space.n
Q = np.zeros((state_size, action_size))

alpha = 0.1
gamma = 0.99
total_episodes = 10000

def update_q_table(state, action, reward, new_state):
	old_value = Q[state, action]
	next_max = max(Q[new_state])
	Q[state, action] = (1-alpha) * old_value + alpha *(reward + (gamma * next_max))
# implementing epsilon-greedy policy

def epsilon_greedy(state):
	if np.random.rand()<epsilon:
		action = env.action_space.sample()   # Explore
	else:
		action = np.argmax(Q[state, :])    # Exploit
	return action

# traning epsilon-greedy agent
epsilon = 0.9
rewards_eps_greedy = []
for episode in range(total_episodes):
	state, info = env.reset()
	terminated = False
	episode_reward = 0
	while not terminated:
		action = epsilon_greedy(state)
		new_state, reward, terminated, truncated, indo = env.step(action)
		update_q_table(state, action, reward, new_state)
		state = new_state
		episode_reward += reward
	rewards_eps_greedy.append(episode_reward)
	

# training decayed epsilon-greedy agent
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
rewards_decay_eps_greedy = []

for episode in range(total_episodes):
	state, info = env.reset()
	terminated = False
	episode_reward = 0
	while not terminated:
		action = epsilon_greedy(state)
		new_state, reward, terminated, truncated, indo = env.step(action)
		episode_reward += reward
		update_q_table(state, action, reward, new_state)
		state = new_state
	rewards_decay_eps_greedy.append(episode_reward)
	epsilon = max(min_epsilon, epsilon * epsilon_decay)


# comparing strategies
avg_eps_greedy = np.mean(rewards_eps_greedy)
avg_decay = np.mean(rewards_decay_eps_greedy)
plt.bar(['Epsilon Greedy', 'Decayed Epsilon Greedy'], [avg_eps_greedy, avg_decay], color=['blue', 'green'])
plt.title('Average Reward per Episode')
plt.ylabel('Average Reward')
plt.savefig('avg_reward_comparison.png')
plt.show()