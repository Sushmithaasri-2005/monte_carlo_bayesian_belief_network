import numpy as np

# Define the probabilities as per the Bayesian Belief Network
P_A = {"yes": 0.8, "no": 0.2}  # Prior for A (Aptitude Skills)
P_C = {"yes": 0.5, "no": 0.5}  # Prior for C (Coding Skills)

# Conditional probabilities for G given A and C
P_G_given_A_C = {
    ("Good", "yes", "yes"): 0.9,
    ("Good", "yes", "no"): 0.7,
    ("Good", "no", "yes"): 0.6,
    ("Good", "no", "no"): 0.3,
    ("OK", "yes", "yes"): 0.1,
    ("OK", "yes", "no"): 0.3,
    ("OK", "no", "yes"): 0.4,
    ("OK", "no", "no"): 0.7,
}

# Conditional probabilities for J (Go for Job) and S (Start a Startup)
P_J_given_G = {
    ("Good", "yes"): 0.8,
    ("Good", "no"): 0.2,
    ("OK", "yes"): 0.2,
    ("OK", "no"): 0.8,
}
P_S_given_G = {
    ("Good", "yes"): 0.7,
    ("Good", "no"): 0.3,
    ("OK", "yes"): 0.3,
    ("OK", "no"): 0.7,
}

# Monte Carlo simulation to estimate conditional probabilities
def monte_carlo_simulation(target_node, evidence, num_samples=10000):
    """
    Perform Monte Carlo simulation to compute the conditional probability of target_node given evidence.

    :param target_node: The node for which to compute the probability (e.g., 'G')
    :param evidence: Dictionary of evidence (e.g., {'A': 'yes', 'C': 'yes'})
    :param num_samples: Number of samples to generate
    :return: Estimated conditional probability distribution for the target node
    """
    samples = []

    for _ in range(num_samples):
        # Sample A and C from their priors
        A = np.random.choice(["yes", "no"], p=[P_A["yes"], P_A["no"]])
        C = np.random.choice(["yes", "no"], p=[P_C["yes"], P_C["no"]])

        # Sample G based on A and C
        probs_G = [P_G_given_A_C[("Good", A, C)], P_G_given_A_C[("OK", A, C)]]
        G = np.random.choice(["Good", "OK"], p=probs_G)

        # Sample J and S based on G
        probs_J = [P_J_given_G[(G, "yes")], P_J_given_G[(G, "no")]]
        J = np.random.choice(["yes", "no"], p=probs_J)

        probs_S = [P_S_given_G[(G, "yes")], P_S_given_G[(G, "no")]]
        S = np.random.choice(["yes", "no"], p=probs_S)

        # Store the sample
        samples.append({"A": A, "C": C, "G": G, "J": J, "S": S})

    # Filter samples matching the evidence
    filtered_samples = [sample for sample in samples if all(sample[key] == value for key, value in evidence.items())]

    # Compute the conditional probability distribution for the target node
    target_counts = {}
    total_filtered = len(filtered_samples)
    for sample in filtered_samples:
        target_value = sample[target_node]
        target_counts[target_value] = target_counts.get(target_value, 0) + 1

    # Normalize to get probabilities
    target_probs = {key: count / total_filtered for key, count in target_counts.items()}
    return target_probs

# Perform inference for G given evidence A=yes, C=yes
target_node = "G"
evidence = {"A": "yes", "C": "yes"}
num_samples = 10000

result = monte_carlo_simulation(target_node, evidence, num_samples)
print(f"Estimated conditional probabilities for {target_node} given evidence {evidence}: {result}")
