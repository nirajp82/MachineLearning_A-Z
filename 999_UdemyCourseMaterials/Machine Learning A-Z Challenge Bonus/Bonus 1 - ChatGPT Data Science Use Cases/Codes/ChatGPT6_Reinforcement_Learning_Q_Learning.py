# Self-Driving Car (Q-Learning)

# Import the libraries
import gym
import numpy as np

# Create the environment
env = gym.make('SelfDrivingCar-v0')

# Set the number of actions
num_actions = env.action_space.n

# Set the number of states
num_states = env.observation_space.n

# Initialize the Q-table
Q = np.zeros((num_states, num_actions))

# Set the learning rate
lr = 0.1

# Set the discount factor
discount_factor = 0.95

# Set the exploration rate
exploration_rate = 0.5

# Set the maximum number of episodes
max_episodes = 1000

# Set the maximum number of steps per episode
max_steps_per_episode = 100

# Train the Q-learning model
for episode in range(max_episodes):
  # Initialize the episode
  state = env.reset()
  done = False
  
  for step in range(max_steps_per_episode):
    # Choose an action
    if np.random.uniform(0, 1) < exploration_rate:
      action = env.action_space.sample()
    else:
      action = np.argmax(Q[state, :])
      
    # Take the action
    next_state, reward, done, info = env.step(action)
    
    # Update the Q-table
    Q[state, action] = Q[state, action] + lr * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
    
    # Set the state to the next state
    state = next_state
    
    # If the episode is finished, break the loop
    if done:
      break
      
# Test the Q-learning model
env.reset()
done = False

while not done:
  action = np.argmax(Q[state, :])
  state, reward, done, info = env.step(action)
  env.render()

# Close the environment
env.close()
