import gym
import gym_minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the custom neural network for the agent
class CustomNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to preprocess observations
def preprocess_observation(obs):
    image = obs['image'].flatten()
    direction = np.array([obs['direction']])
    mission = np.zeros(1)  # Dummy encoding for the mission
    return np.concatenate([image, direction, mission])

# Custom training method
def train_agent(env, model, optimizer, num_episodes=1000):
    criterion = nn.MSELoss()
    for episode in range(num_episodes):
        obs = env.reset()[0]
        obs = preprocess_observation(obs)
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_probs = model(obs_tensor)
            action = torch.argmax(action_probs).item()
            results = env.step(action)
            next_obs, reward, done, _, _ = results
            next_obs = preprocess_observation(next_obs)
            total_reward += reward

            # Calculate loss and update model
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
            target = reward + 0.99 * torch.max(model(next_obs_tensor)).item()
            target_tensor = action_probs.clone().detach()
            target_tensor[action] = target
            loss = criterion(action_probs, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs

        print(f'Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}')

# Main function
def main():
    env = gym.make('MiniGrid-Empty-Random-6x6-v0')  # Choose the appropriate Minigrid environment
    obs = env.reset()[0]
    input_dim = len(preprocess_observation(obs))
    output_dim = env.action_space.n

    model = CustomNet(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_agent(env, model, optimizer)

if __name__ == "__main__":
    main()
