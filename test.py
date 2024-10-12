import numpy as np

# Load the existing attack categories from the npy file
attack_categories = np.load("le2_classes.npy", allow_pickle=True)

# Display the current categories
print("Current categories:", attack_categories)

# Modify the category names
# Example modifications (change these to your desired names)
new_categories = {
    'Analysis': 'Generic',
    'Backdoor': 'worms',
    'DoS': 'Analysis',
    'Exploits': 'Fuzzer',
    'Fuzzers': 'Exploits',
    'Generic': 'Normal',
    'Normal': 'Backdoor',
    'Reconnaissance': 'Reconnaissance',
    'Worms': 'Dos'
}

# Update the category names
for i, category in enumerate(attack_categories):
    if category in new_categories:
        attack_categories[i] = new_categories[category]

# Display the updated categories
print("Updated categories:", attack_categories)

# Save the updated categories back to an npy file
np.save("classes.npy", attack_categories)