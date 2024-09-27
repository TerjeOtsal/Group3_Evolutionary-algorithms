# Import necessary libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
import time  # For controlling the rendering speed

# Create the Taxi-v3 environment
# This environment simulates a grid-based taxi problem where the agent needs to pick up a passenger
# and drop them off at a specified location while avoiding illegal actions.
env = gym.make("Taxi-v3")

# Ensure compatibility with different versions of Gym where the reset method returns additional information
# Depending on the version of Gym, `env.reset()` may return only the state or a tuple (state, info).
try:
    state = env.reset()[0]  # For newer gym versions, env.reset() returns a tuple (state, info)
except:
    state = env.reset()  # For older gym versions, env.reset() returns just the state

# Initialize the Q-table with zeros
# Q-table stores the Q-values (expected future rewards) for each state-action pair
# Shape: (number of states, number of actions)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Define the hyperparameters for Q-learning
alpha = 0.1   # Learning rate: How much we update the Q-value at each step
gamma = 0.8   # Discount factor: How much we consider future rewards
epsilon = 0.1  # Exploration-exploitation trade-off: Probability of choosing a random action

# Define the number of episodes for training
num_episodes = 10000

# Create a list to hold total rewards for each episode
# This helps in analyzing the training progress over time.
total_rewards = []

# Training the Q-learning agent over the specified number of episodes
for episode in range(num_episodes):
    # Reset the environment to the initial state at the start of each episode
    try:
        state = env.reset()[0]  # For newer gym versions, reset() returns (state, info)
    except:
        state = env.reset()  # For older gym versions

    done = False  # This variable keeps track of whether the episode is complete
    total_reward = 0  # Initialize total reward for the episode

    while not done:
        # Choose an action based on the epsilon-greedy strategy
        # With probability `epsilon`, take a random action (exploration)
        # Otherwise, take the action with the highest Q-value for the current state (exploitation)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: choose a random action
        else:
            action = np.argmax(q_table[state])  # Exploit: choose the action with max Q-value for the current state

        # Take the chosen action and observe the result
        try:
            next_state, reward, done, truncated, _ = env.step(action)  # Updated to handle the truncated variable
        except:
            next_state, reward, done, _ = env.step(action)  # For older versions

        # Update the Q-value using the Q-learning formula:
        # Q(state, action) = Q(state, action) + alpha * (reward + gamma * max(Q(next_state, all_actions)) - Q(state, action))
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        # Accumulate the total reward for this episode
        total_reward += reward

        # Move to the next state
        state = next_state

    # Store the total reward for this episode to analyze learning progress
    total_rewards.append(total_reward)

    # Print the average score for the last 20 episodes every 20 episodes
    if (episode + 1) % 20 == 0:
        average_last_20 = np.mean(total_rewards[-20:])  # Calculate the average of the last 20 episodes
        print(f"Episode {episode + 1}/{num_episodes}: Average reward for last 20 episodes = {average_last_20}")

# Training completed
print("Training completed.")

# Plot the total rewards over time to visualize the learning progress
plt.plot(range(num_episodes), total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Rewards Over Time")
plt.show()

# Function to evaluate the performance of the trained policy (Q-table)
# This function calculates the average reward over multiple episodes using the learned policy
def evaluate_policy(q_table, num_episodes=100):
    total_rewards = 0
    for episode in range(num_episodes):
        try:
            state = env.reset()[0]  # For newer gym versions
        except:
            state = env.reset()  # For older gym versions

        done = False
        while not done:
            # Choose the action with the highest Q-value for the current state
            action = np.argmax(q_table[state])
            try:
                next_state, reward, done, truncated, _ = env.step(action)  # Updated to handle the truncated variable
            except:
                next_state, reward, done, _ = env.step(action)  # For older versions
            total_rewards += reward  # Accumulate rewards for each episode
            state = next_state
    # Return the average reward across all episodes
    return total_rewards / num_episodes

# Evaluate the performance of the trained Q-learning policy
q_learning_performance = evaluate_policy(q_table)
print(f"Average performance of the trained Q-learning agent: {q_learning_performance}")

# Evaluate a random policy for comparison
# Here, we create a Q-table with all zeros (no learning), which represents a random policy
random_performance = evaluate_policy(np.zeros([env.observation_space.n, env.action_space.n]))
print(f"Average performance of the random policy: {random_performance}")

# Function to show the trained agent working on a single episode
# This function visually demonstrates the trained agent in the environment
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
    env.render()  # Render the initial state of the environment
    print("Starting demonstration...")

    # Run until the agent successfully drops off the passenger
    while not done:
        # Choose the action with the highest Q-value for the current state
        action = np.argmax(q_table[state])
        try:
            next_state, reward, done, truncated, _ = env.step(action)  # Updated to handle the truncated variable
        except:
            next_state, reward, done, _ = env.step(action)  # For older versions

        # Accumulate total reward for demonstration
        total_reward += reward

        # Render the environment to visualize the agent's actions
        env.render()
        time.sleep(0.5)  # Add a delay to make the simulation slower and easier to watch

        # Move to the next state
        state = next_state

    print(f"Total reward: {total_reward}")  # Print the total reward achieved by the agent
    env.close()  # Close the environment

# Show the trained agent working on a single episode
show_trained_agent(env, q_table)
