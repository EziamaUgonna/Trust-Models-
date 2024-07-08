class GameTheoreticTrust:
    def __init__(self):
        self.trust_scores = {}

    def update_trust(self, node_id, action, reward):
        if node_id not in self.trust_scores:
            self.trust_scores[node_id] = 0
        # Simple update based on action and reward
        self.trust_scores[node_id] += reward if action == "cooperate" else -reward

    def trust_score(self, node_id):
        return self.trust_scores.get(node_id, 0)

# Example usage
gt_trust = GameTheoreticTrust()
gt_trust.update_trust("Node1", "cooperate", 1)
gt_trust.update_trust("Node1", "defect", 0.5)
print("Trust Score of Node1:", gt_trust.trust_score("Node1"))
