# run_trained_agent.py

import gym
import numpy as np
import time

# Load the trained Q-table from the file
q_table = np.load("trained_q_table.npy")
print("Loaded trained Q-table.")

# Initialize the Taxi-v3 environment with render mode set to "ansi"
env = gym.make("Taxi-v3", render_mode="ansi")

# Number of episodes to run the trained agent
num_episodes = 5

# Run the trained agent for the specified number of episodes
for episode in range(num_episodes):
    print(f"\nStarting episode {episode + 1}/{num_episodes}\n")
    try:
        state = env.reset()[0]  # Reset environment and get initial state
    except:
        state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select the best action based on the trained Q-table
        action = np.argmax(q_table[state])

        # Take action and observe the result
        try:
            next_state, reward, done, truncated, _ = env.step(action)
        except:
            next_state, reward, done, _ = env.step(action)
        
        total_reward += reward
        state = next_state

        # Render the environment for each step (no need to specify mode as it was set on env creation)
        print(env.render())
        time.sleep(0.5)  # Slow down for observation

        # Check if the episode should end (if drop-off is successful)
        if done or (reward == 20):  # Reward of 20 in Taxi-v3 typically indicates a successful drop-off
            print("Passenger successfully dropped off. Ending episode.")
            done = True

    print(f"Total reward for episode {episode + 1}: {total_reward}")

# Close the environment
env.close()
print("\nAll episodes completed.")
