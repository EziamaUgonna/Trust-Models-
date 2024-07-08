import numpy as np

# Function to compute cosine similarity
def cosine_similarity(opinions, recommendations):
    dot_product = np.dot(opinions, recommendations)
    norm_opinions = np.linalg.norm(opinions)
    norm_recommendations = np.linalg.norm(recommendations)
    if norm_opinions == 0 or norm_recommendations == 0:
        return 0.0
    return dot_product / (norm_opinions * norm_recommendations)

# Function to compute entropy
def entropy(prob, base=2):
    if prob == 0:
        return 0
    return -prob * np.log(prob) / np.log(base)

# Function to compute trust
def compute_trust(similarity, entropy_similarity, entropy_recommendation, prev_trust, theta=0.5):
    trust = theta * (1 - (entropy_similarity + entropy_recommendation)) + (1 - theta) * prev_trust
    return min(max(trust, 0), 1)

# Example data
E_opinions = np.array([1, 0, 1, 0, 1])
recommendations = [
    np.array([1, 1, 1, 0, 1]),  # s1
    np.array([1, 0, 1, 0, 0]),  # s2
    np.array([0, 0, 0, 1, 1]),  # s3
]

# Compute similarity and entropy for each recommender
similarities = []
entropies_similarity = []
entropies_recommendation = []

for rec in recommendations:
    sim = cosine_similarity(E_opinions, rec)
    similarities.append(sim)
    entropies_similarity.append(entropy(sim))
    
    # Compute entropy for the recommendation itself
    p_recommendation = [np.mean(rec == value) for value in [0, 1]]
    entropy_recommendation = sum(entropy(p) for p in p_recommendation)
    entropies_recommendation.append(entropy_recommendation)

# Initialize trust values
initial_trust = [0.5 for _ in recommendations]  # Assume a neutral starting trust
theta = 0.5  # Weight parameter

# Update trust values
trust_values = []
for i in range(len(recommendations)):
    trust = compute_trust(
        similarities[i], 
        entropies_similarity[i], 
        entropies_recommendation[i], 
        initial_trust[i], 
        theta
    )
    trust_values.append(trust)

print("Similarity Scores:", similarities)
print("Entropies of Similarity:", entropies_similarity)
print("Entropies of Recommendations:", entropies_recommendation)
print("Trust Values:", trust_values)
