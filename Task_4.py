# Import necessary libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
import time  # For controlling the rendering speed

# Create the Taxi-v3 environment
env = gym.make("Taxi-v3")


# Ensure compatibility with different versions of Gym where the reset method returns additional information
try:
    state = env.reset()[0]  # For newer gym versions, env.reset() returns a tuple (state, info)
except:
    state = env.reset()  # For older gym versions, env.reset() returns just the state

# Initialize the Q-table with zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Define the hyperparameters
alpha = 0.1    # Learning rate
gamma = 0.6    # Discount factor
epsilon = 0.1  # Exploration-exploitation trade-off

# Define the number of episodes
num_episodes = 500

# Create a list to hold total rewards for each episode
total_rewards = []

# Training the Q-learning agent
for episode in range(num_episodes):
    try:
        state = env.reset()[0]  # For newer gym versions, reset() returns (state, info)
    except:
        state = env.reset()  # For older gym versions

    done = False
    total_reward = 0

    while not done:
        # Exploration vs. Exploitation
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: choose a random action
        else:
            action = np.argmax(q_table[state])  # Exploit: choose the action with max Q-value for the current state

        # Take the action and observe the result
        try:
            next_state, reward, done, truncated, _ = env.step(action)  # Updated to handle the truncated variable
        except:
            next_state, reward, done, _ = env.step(action)  # For older versions

        # Update the Q-value using the Q-learning formula
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        # Update the total reward
        total_reward += reward

        # Move to the next state
        state = next_state

    # Append total reward of the episode to the rewards list
    total_rewards.append(total_reward)

# Training completed
print("Training completed.")

# Plot the total rewards over time
plt.plot(range(num_episodes), total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Rewards Over Time")
plt.show()

# Function to evaluate the performance of a policy
def evaluate_policy(q_table, num_episodes=100):
    total_rewards = 0
    for episode in range(num_episodes):
        try:
            state = env.reset()[0]  # For newer gym versions
        except:
            state = env.reset()  # For older gym versions

        done = False
        while not done:
            action = np.argmax(q_table[state])  # Choose the action with max Q-value for the current state
            try:
                next_state, reward, done, truncated, _ = env.step(action)  # Updated to handle the truncated variable
            except:
                next_state, reward, done, _ = env.step(action)  # For older versions
            total_rewards += reward
            state = next_state
    return total_rewards / num_episodes

# Evaluate the trained Q-learning policy
q_learning_performance = evaluate_policy(q_table)
print(f"Average performance of the trained Q-learning agent: {q_learning_performance}")

# Evaluate a random policy
random_performance = evaluate_policy(np.zeros([env.observation_space.n, env.action_space.n]))  # A Q-table with zero values
print(f"Average performance of the random policy: {random_performance}")

# Function to show the trained agent working on a single episode
def show_trained_agent(env, q_table):
    """
    Function to show the trained agent working on a single episode.
    This function renders each step of the trained agent in action.
    """
    try:
        state = env.reset()[0]  # For newer gym versions
    except:
        state = env.reset()  # For older gym versions

    done = False
    total_reward = 0
    env.render()
    print("Starting demonstration...")

    # Run until the agent successfully drops off the passenger
    while not done:
        action = np.argmax(q_table[state])  # Choose the action with max Q-value for the current state
        try:
            next_state, reward, done, truncated, _ = env.step(action)  # Updated to handle the truncated variable
        except:
            next_state, reward, done, _ = env.step(action)  # For older versions

        # Update the total reward
        total_reward += reward

        # Render the environment to visualize
        env.render()
        time.sleep(0.5)  # Add a delay to make the simulation slower and easier to watch

        # Move to the next state
        state = next_state

    print(f"Total reward: {total_reward}")
    env.close()

# Show the trained agent working on a single episode
show_trained_agent(env, q_table)
