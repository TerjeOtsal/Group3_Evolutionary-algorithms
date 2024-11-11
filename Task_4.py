# Import necessary libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("Initializing environment and Q-table...")
# Initialize the Taxi-v3 environment with render mode for training
env = gym.make("Taxi-v3", render_mode="ansi")

# Initialize the Q-table with zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Define hyperparameters for Q-learning
alpha = 0.1           # Learning rate
gamma = 0.8           # Discount factor
epsilon = 1.0         # Initial exploration rate
epsilon_decay = 0.995 # Decay rate for epsilon
min_epsilon = 0.01    # Minimum exploration rate
num_episodes = 200000  # Maximum episodes for training

# Tracking rewards and parameters for analysis
total_rewards = []        # Rewards per episode
average_reward_history = [] # Average rewards over 10-episode intervals
epsilon_values = []       # Epsilon values over episodes
patience_limit = 10000       # Early stopping if no improvement over 10000 intervals
patience_count = 0        # Counter for patience
early_stopped = False     # Flag for early stopping

print("Starting Q-learning training loop...")
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

print("Training completed. Closing environment...")
# Close the training environment after training
env.close()

print("Saving trained Q-table to file...")
# Save the trained Q-table to a file
np.save("trained_q_table.npy", q_table)
print("Trained Q-table saved to 'trained_q_table.npy'.")

# Use Seaborn style for visualization
sns.set(style="whitegrid", font_scale=1.1)

# Plot 1: Total Rewards Over Time (every 100th episode)
print("Plotting total rewards over time...")
plt.figure(figsize=(10, 5))
plt.plot(range(0, len(total_rewards), 100), total_rewards[::100], color='royalblue', linewidth=1.5)
plt.xlabel("Episode (every 100th)")
plt.ylabel("Total Reward")
plt.title("Total Rewards Over Time")
plt.grid(True)
plt.show()

# Plot 2: Average Reward for Every 10 Episodes
print("Plotting average reward for every 10 episodes...")
plt.figure(figsize=(10, 5))
plt.plot(range(10, len(average_reward_history) * 10 + 1, 10), average_reward_history, color='tomato', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Average Reward (per 10 episodes)")
plt.title("Average Reward Over Every 10 Episodes")
plt.grid(True)
plt.show()

# Plot 3: Epsilon Decay Over Time
print("Plotting epsilon decay over time...")
plt.figure(figsize=(10, 5))
plt.plot(range(len(epsilon_values)), epsilon_values, color='purple', linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel("Epsilon Value")
plt.title("Epsilon Decay Over Time")
plt.grid(True)
plt.show()

# Evaluation function to assess the trained policy
def evaluate_policy(q_table, num_episodes=100, max_steps=200):
    """
    Evaluates the trained policy over multiple episodes to calculate average reward.
    """
    print("Initializing environment for evaluation...")
    
    # Initialize a new environment instance for evaluation
    eval_env = gym.make("Taxi-v3", render_mode="ansi")
    total_rewards = []

    for episode in range(num_episodes):
        print(f"Evaluation episode {episode + 1} - Resetting environment...")
        
        # Reset the environment and capture the initial state
        try:
            state = eval_env.reset()[0] if isinstance(eval_env.reset(), tuple) else eval_env.reset()
            print(f"State after reset for episode {episode + 1}: {state}")
        except Exception as e:
            print(f"Error resetting environment for episode {episode + 1}: {e}")
            eval_env.close()
            return []

        done = False
        episode_reward = 0
        step_count = 0  # Initialize the step counter

        # Run the episode with a maximum step limit to avoid infinite loops
        while not done and step_count < max_steps:
            # Choose action based on the Q-table
            action = np.argmax(q_table[state])

            try:
                # Take a step in the environment using the chosen action
                result = eval_env.step(action)
                next_state, reward, done = result[:3]  # Support environments without truncated return

                episode_reward += reward
                state = next_state
                step_count += 1  # Increment the step counter

                # Print debug information for each step
                print(f"Episode {episode + 1}, Step {step_count}: Action={action}, Reward={reward}, Next State={next_state}, Done={done}")

            except Exception as e:
                print(f"Error during step in episode {episode + 1}: {e}")
                eval_env.close()
                return []

        # Check if the episode was forced to terminate due to reaching max_steps
        if step_count >= max_steps:
            print(f"Episode {episode + 1} terminated after reaching max steps limit of {max_steps}")

        # Store total reward for this episode
        total_rewards.append(episode_reward)
        print(f"Total reward for evaluation episode {episode + 1}: {episode_reward}")

    # Close the evaluation environment
    print("Evaluation completed. Closing evaluation environment...")
    eval_env.close()
    return total_rewards



# Gather rewards for histogram after training
print("Evaluating policy...")
evaluation_rewards = evaluate_policy(q_table)

# Plot 4: Histogram of Total Rewards per Episode After Training
print("Plotting histogram of total rewards per episode after training...")
plt.figure(figsize=(10, 5))
plt.hist(evaluation_rewards, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Total Reward per Episode")
plt.ylabel("Frequency")
plt.title("Histogram of Total Rewards per Episode After Training")
plt.grid(True)
plt.show()

def show_trained_agent(q_table, max_steps=20):
    """
    Shows the agent’s behavior in a single episode using the trained Q-table policy.
    If the agent exceeds max_steps, the environment resets to avoid getting stuck.
    """
    print("Initializing environment for demonstration...")
    # Initialize a new environment instance for demonstration
    demo_env = gym.make("Taxi-v3", render_mode="ansi")
    
    def reset_environment():
        """Resets the environment and returns the initial state."""
        try:
            state = demo_env.reset()[0]
        except:
            state = demo_env.reset()
        return state

    # Initial setup
    state = reset_environment()
    done = False
    total_reward = 0
    step_count = 0  # Initialize step counter
    
    print("Starting demonstration...\n")
    while not done:
        action = np.argmax(q_table[state])
        
        # Take a step and handle potential errors in step execution
        try:
            result = demo_env.step(action)
            next_state, reward, done = result[:3]
            total_reward += reward
            print(demo_env.render())
            time.sleep(0.5)
            state = next_state
            step_count += 1
        except Exception as e:
            print(f"Error during demonstration step: {e}")
            demo_env.close()
            return

        # Reset the environment if the agent is stuck in a loop
        if step_count >= max_steps:
            print(f"Agent exceeded max steps ({max_steps}), resetting environment...")
            state = reset_environment()
            total_reward = 0  # Reset reward for the new episode
            step_count = 0    # Reset step counter
    
    # Close the demonstration environment
    print("Demonstration completed. Closing demonstration environment...")
    demo_env.close()
    print(f"\nTotal reward: {total_reward}")

# Show the trained agent in action
print("Demonstrating trained agent's behavior...")
show_trained_agent(q_table)
print("Script completed.")
