import numpy as np

data = np.load("./quickdraw_dataset_transformed/cat_actions.npz", allow_pickle=True)

print(data.files)
# ['actions', 'words', 'key_ids']

actions = data["actions"]   # list-like array of (T_i, 4) matrices
words   = data["words"]
key_ids = data["key_ids"]

# Example: inspect first trajectory
print("Category:", words[0])
print("Key:", key_ids[0])
print("Action shape:", actions[0].shape)
print(actions[0])
