import matplotlib.pyplot as plt
import seaborn as sns

# Extract a batch of features
batch = next(iter(data.test_dataloader))
features = batch['features'].numpy()

# Flatten embeddings
features = features.reshape(features.shape[0], -1)

# Plot distributions
sns.heatmap(features, cmap="viridis")
plt.title("Feature Distribution")
plt.show()
