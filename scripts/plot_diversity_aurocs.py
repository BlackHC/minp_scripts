#%%
import blackhc.project.script
import pickle
import pandas as pd
import numpy as np

#%%
# Load diversity_scores.pkl
metrics = pickle.load(open("diversity_scores.pkl", "rb"))

#%%
# Convert to dataframe
df = pd.DataFrame(metrics).T
#%% 
# Compute average correct_entropy and wrong_entropy for each metric
df["avg_correct_entropy"] = df["correct_entropy"].apply(lambda x: np.mean(x).item())
df["avg_wrong_entropy"] = df["wrong_entropy"].apply(lambda x: np.mean(x).item())
df["is_min_p"] = df["options"].apply(lambda x: x.get("options.min_p") is not None)
#%%
normalized_options = pd.json_normalize(df["options"])
normalized_options.index = df.index
# Add normalized_options to df
df = pd.concat([df, normalized_options], axis=1, ignore_index=False)

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')

fig, ax = plt.subplots(figsize=(6, 6/1.6))

unique_temperatures = df["options.temperature"].unique()
temperature_norm = plt.Normalize(unique_temperatures.min(), unique_temperatures.max())
color_map = plt.cm.viridis

def plot_sorted_by_diversity(data, color, label, marker):
    points = data[["accuracy", "avg_correct_entropy"]].values
    points = np.array(points)
    points = points[points[:, 1].argsort()]
    ax.plot(points[:, 0], points[:, 1], c=color)
    ax.plot([], c=color, linestyle='-', label=label, marker=marker)
    
for is_min_p, group_df in reversed(list(df.groupby("is_min_p"))):
    marker = 'o' if is_min_p else '^'
    label = 'Min-p' if is_min_p else 'Top-p'
    color = 'C0' if is_min_p else 'C1'
    
    plot_sorted_by_diversity(group_df, color, label, marker)
    
    scatter = ax.scatter(group_df["accuracy"], group_df["avg_correct_entropy"],
                         c=group_df["options.temperature"], cmap=color_map, norm=temperature_norm,
                         marker=marker, s=100, alpha=1.0, zorder=10)
     
ax.set_xlabel("Accuracy")
ax.set_ylabel("Avg. Entropy of Correct Predictions (\"Creativity\")")
ax.set_title("Comparison of Min-p and Top-p: Accuracy vs Creativity")
ax.legend()

cbar = fig.colorbar(scatter)
cbar.set_label("Temperature")

plt.tight_layout()
plt.show()
# %%
