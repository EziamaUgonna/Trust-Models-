import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepRLTrustModel:
    def __init__(self, num_nodes, alpha=0.01, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=2000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(num_nodes, num_nodes).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_nodes)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_q_network(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).to(self.device)
            action_tensor = torch.LongTensor([action]).to(self.device)
            done_tensor = torch.FloatTensor([done]).to(self.device)
            
            current_q = self.q_network(state_tensor)[action_tensor]
            max_next_q = torch.max(self.q_network(next_state_tensor))
            target_q = reward_tensor + (self.gamma * max_next_q * (1 - done_tensor))
            
            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_with_data(self, data, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            for state, action, reward, next_state, done in data:
                self.store_transition(state, action, reward, next_state, done)
                self.update_q_network(batch_size)
            
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_trust(self, node):
        state = np.zeros(self.num_nodes)
        state[node] = 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.max(q_values).item()

# Example usage with custom data
num_nodes = 5
rl_trust_model = DeepRLTrustModel(num_nodes)

# Custom interaction data: (state, action, reward, next_state, done)
interaction_data = [
    (np.eye(num_nodes)[0], 1, 1, np.eye(num_nodes)[2], False),
    (np.eye(num_nodes)[1], 2, -1, np.eye(num_nodes)[3], False),
    (np.eye(num_nodes)[2], 3, 1, np.eye(num_nodes)[4], False),
    (np.eye(num_nodes)[3], 4, -1, np.eye(num_nodes)[0], False),
    (np.eye(num_nodes)[4], 0, 1, np.eye(num_nodes)[1], False),
    # Add more interactions as needed
]

# Train the model with custom data
rl_trust_model.train_with_data(interaction_data, epochs=500)

# Get trust values
for node in range(num_nodes):
    print(f"Trust value for node {node}: {rl_trust_model.get_trust(node)}")

# Print the Q-Table for reference
print("\nQ-Network Weights:")
for param in rl_trust_model.q_network.parameters():
    print(param.data)
