import numpy as np
import random

class RLTrustModel:
    def __init__(self, num_nodes, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize the Reinforcement Learning Trust Model.

        Parameters:
        num_nodes (int): Number of nodes in the network.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration factor.
        """
        self.num_nodes = num_nodes
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.q_table = np.zeros((num_nodes, num_nodes))  # Q-Table for trust values

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy strategy.

        Parameters:
        state (int): Current state (node).

        Returns:
        int: Selected action (node to interact with).
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_nodes - 1)  # Explore: choose random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: choose action with max Q-value

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table based on the observed reward.

        Parameters:
        state (int): Current state (node).
        action (int): Action taken (node interacted with).
        reward (float): Observed reward.
        next_state (int): Next state (node).
        """
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - predict)

    def get_trust(self, node):
        """
        Get the trust value for a specific node.

        Parameters:
        node (int): Node to get the trust value for.

        Returns:
        float: Trust value of the node.
        """
        return np.max(self.q_table[node])

    def train_with_data(self, data):
        """
        Train the model using custom interaction data.

        Parameters:
        data (list): List of interactions represented as tuples (state, action, reward, next_state).
        """
        for state, action, reward, next_state in data:
            self.update_q_table(state, action, reward, next_state)

# Example usage with custom data
num_nodes = 5
rl_trust_model = RLTrustModel(num_nodes)

# Custom interaction data: (state, action, reward, next_state)
interaction_data = [
    (0, 1, 1, 2),
    (1, 2, -1, 3),
    (2, 3, 1, 4),
    (3, 4, -1, 0),
    (4, 0, 1, 1),
    # Add more interactions as needed
]

# Train the model with custom data
rl_trust_model.train_with_data(interaction_data)

# Get trust values
for node in range(num_nodes):
    print(f"Trust value for node {node}: {rl_trust_model.get_trust(node)}")

# Print the Q-table for reference
print("\nQ-Table:")
print(rl_trust_model.q_table)
