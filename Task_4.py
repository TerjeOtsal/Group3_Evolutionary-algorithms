# Import necessary libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Initialize the Taxi-v3 environment with render mode
env = gym.make("Taxi-v3", render_mode="ansi")

# Initialize the Q-table with zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Define hyperparameters for Q-learning
alpha = 0.1           # Learning rate
gamma = 0.8           # Discount factor
epsilon = 1.0         # Initial exploration rate
epsilon_decay = 0.995 # Decay rate for epsilon
min_epsilon = 0.01    # Minimum exploration rate
num_episodes = 10000  # Maximum episodes for training

# Tracking rewards and parameters for analysis
total_rewards = []        # Rewards per episode
average_reward_history = [] # Average rewards over 10-episode intervals
epsilon_values = []       # Epsilon values over episodes
patience_limit = 50       # Early stopping if no improvement over 20 intervals
patience_count = 0        # Counter for patience
early_stopped = False     # Flag for early stopping

# Q-learning training loop with early stopping
for episode in range(num_episodes):
    # Reset environment at the start of each episode
    try:
        state = env.reset()[0]
    except:
        state = env.reset()
    
    done = False
    total_reward = 0
    
    # Run the episode until completion
    while not done:
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(q_table[state])  # Exploitation

        # Take action and observe result
        try:
            next_state, reward, done, truncated, _ = env.step(action)
        except:
            next_state, reward, done, _ = env.step(action)

        # Q-value update
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        total_reward += reward
        state = next_state

    # Track total reward and epsilon decay
    total_rewards.append(total_reward)
    epsilon_values.append(epsilon)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Check for average reward every 10 episodes
    if (episode + 1) % 10 == 0:
        average_reward = np.mean(total_rewards[-10:])
        average_reward_history.append(average_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Average reward for last 10 episodes = {average_reward}")

        # Early stopping if no improvement in last 20 checks
        if len(average_reward_history) > 1 and average_reward <= max(average_reward_history[:-1]):
            patience_count += 1
            if patience_count >= patience_limit:
                print("Early stopping activated due to lack of improvement.")
                early_stopped = True
                break
        else:
            patience_count = 0  # Reset patience if improvement is found

# Close the environment after training
env.close()

print("Training completed.")

# Save the trained Q-table to a file
np.save("trained_q_table.npy", q_table)
print("Trained Q-table saved to 'trained_q_table.npy'.")

# Use Seaborn style for visualization
sns.set(style="whitegrid", font_scale=1.1)

# Plot 1: Total Rewards Over Time (every 100th episode)
plt.figure(figsize=(10, 5))
plt.plot(range(0, len(total_rewards), 100), total_rewards[::100], color='royalblue', linewidth=1.5)
plt.xlabel("Episode (every 100th)")
plt.ylabel("Total Reward")
plt.title("Total Rewards Over Time")
plt.grid(True)
plt.show()

# Plot 2: Average Reward for Every 10 Episodes
plt.figure(figsize=(10, 5))
plt.plot(range(10, len(average_reward_history) * 10 + 1, 10), average_reward_history, color='tomato', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Average Reward (per 10 episodes)")
plt.title("Average Reward Over Every 10 Episodes")
plt.grid(True)
plt.show()

# Plot 3: Epsilon Decay Over Time
plt.figure(figsize=(10, 5))
plt.plot(range(len(epsilon_values)), epsilon_values, color='purple', linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel("Epsilon Value")
plt.title("Epsilon Decay Over Time")
plt.grid(True)
plt.show()

# Evaluation function to assess the trained policy
def evaluate_policy(q_table, num_episodes=100):
    """
    Evaluates the trained policy over multiple episodes to calculate average reward.
    """
    total_rewards = []
    for episode in range(num_episodes):
        try:
            state = env.reset()[0]
        except:
            state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = np.argmax(q_table[state])
            try:
                next_state, reward, done, truncated, _ = env.step(action)
            except:
                next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return total_rewards

# Reinitialize the environment for evaluation
env = gym.make("Taxi-v3", render_mode="ansi")

# Gather rewards for histogram after training
evaluation_rewards = evaluate_policy(q_table)

# Plot 4: Histogram of Total Rewards per Episode After Training
plt.figure(figsize=(10, 5))
plt.hist(evaluation_rewards, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Total Reward per Episode")
plt.ylabel("Frequency")
plt.title("Histogram of Total Rewards per Episode After Training")
plt.grid(True)
plt.show()

# Function to demonstrate the trained agent's actions in a single episode
def show_trained_agent(env, q_table):
    """
    Shows the agent’s behavior in a single episode using the trained Q-table policy.
    """
    try:
        state = env.reset()[0]
    except:
        state = env.reset()
    done = False
    total_reward = 0
    print("Starting demonstration...\n")
    while not done:
        action = np.argmax(q_table[state])
        try:
            next_state, reward, done, truncated, _ = env.step(action)
        except:
            next_state, reward, done, _ = env.step(action)
        total_reward += reward
        print(env.render(mode="ansi"))
        time.sleep(0.5)
        state = next_state
    print(f"\nTotal reward: {total_reward}")

# Show the trained agent in action
show_trained_agent(env, q_table)
