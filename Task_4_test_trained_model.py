# Import necessary libraries
import gym
import numpy as np
import time

# Load the trained Q-table
q_table = np.load("q_table.npy")
print("Q-table loaded successfully.")

# Create the Taxi-v3 environment
env = gym.make("Taxi-v3")

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
