import skfuzzy as fuzz
import numpy as np

class FuzzyTrust:
    def __init__(self):
        # Define fuzzy sets
        self.x_trust = np.arange(0, 1.1, 0.1)
        self.low = fuzz.trimf(self.x_trust, [0, 0, 0.5])
        self.medium = fuzz.trimf(self.x_trust, [0, 0.5, 1])
        self.high = fuzz.trimf(self.x_trust, [0.5, 1, 1])

    def evaluate_trust(self, interaction):
        # Fuzzy membership
        low_membership = fuzz.interp_membership(self.x_trust, self.low, interaction)
        medium_membership = fuzz.interp_membership(self.x_trust, self.medium, interaction)
        high_membership = fuzz.interp_membership(self.x_trust, self.high, interaction)
        
        return {
            "low": low_membership,
            "medium": medium_membership,
            "high": high_membership
        }

# Example usage
fuzzy_trust = FuzzyTrust()
trust_levels = fuzzy_trust.evaluate_trust(0.7)
print("Trust Levels:", trust_levels)
