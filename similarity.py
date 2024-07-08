import numpy as np

def jaccard_similarity(G, Rk):
    """Calculate the Jaccard similarity between sets G and Rk."""
    intersection = np.sum(np.logical_and(G, Rk))
    union = np.sum(np.logical_or(G, Rk))
    return intersection / union if union != 0 else 0

def entropy(p, A):
    """Calculate entropy given probability p and cardinality A."""
    return -p * np.log2(p) if p > 0 else 0

def trust_update(Tk_prev, Hsk, HRk, theta):
    """Update the trust value based on the entropy values."""
    return theta * (1 - (Hsk + HRk)) + (1 - theta) * Tk_prev

def simulate_trust_model(G, recommendations, A, theta, num_messages, q, p_malicious):
    """Simulate the trust model and calculate TPR and FPR."""
    num_nodes = len(recommendations)
    trust_values = np.zeros(num_nodes)
    true_positives = 0
    false_positives = 0
    total_malicious = 0
    total_honest = 0

    for msg in range(num_messages):
        for i, Rk in enumerate(recommendations):
            p_sk = jaccard_similarity(G, Rk)
            H_sk = entropy(p_sk, A)
            
            p_rk = [np.mean([Rk[j] for Rk in recommendations]) for j in range(A)]
            H_Rk = np.sum([entropy(p, A) for p in p_rk])
            
            trust_values[i] = trust_update(trust_values[i], H_sk, H_Rk, theta)
            
            # For simulation purposes, assume first half nodes are honest and second half are malicious
            if i < num_nodes // 2:
                if trust_values[i] < 0.5:
                    false_positives += 1
                total_honest += 1
            else:
                if trust_values[i] >= 0.5:
                    true_positives += 1
                total_malicious += 1

    TPR = true_positives / total_malicious if total_malicious > 0 else 0
    FPR = false_positives / total_honest if total_honest > 0 else 0
    return TPR, FPR, trust_values

# Parameters
A = 100  # Cardinality
theta = 0.5  # Weight parameter
num_messages = 100
num_nodes = 10
q = 30
p_malicious = 0.2

# Initialize opinions and recommendations (randomly for demonstration)
np.random.seed(42)
G = np.random.randint(0, 2, A)
recommendations = [np.random.randint(0, 2, A) for _ in range(num_nodes // 2)]
recommendations += [np.random.choice([0, 1], A, p=[1 - p_malicious, p_malicious]) for _ in range(num_nodes // 2, num_nodes)]

# Simulate the trust model
TPR, FPR, trust_values = simulate_trust_model(G, recommendations, A, theta, num_messages, q, p_malicious)

print(f"True Positive Rate (TPR): {TPR:.2f}")
print(f"False Positive Rate (FPR): {FPR:.2f}")
print(f"Trust Values: {trust_values}")
