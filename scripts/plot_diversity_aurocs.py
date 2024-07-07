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
#%%
# Plot accuracy over avg_correct_entropy
import matplotlib.pyplot as plt

df["linestyle"] = df["is_min_p"].map({True: "-", False: "--"})

unique_temperatures = df["options.temperature"].unique()
unique_min_p_values = df[df["is_min_p"]]["options.min_p"].unique()
unique_top_p_values = df[~df["is_min_p"]]["options.top_p"].unique()

for (is_min_p, temperature), sub_df in df.groupby(["is_min_p", "options.temperature"]):
    linestyle = "-" if is_min_p else "--"
    temperature_index = unique_temperatures.tolist().index(temperature)
    color = f"C{temperature_index}"
    # label = f"MinP Temperature: {temperature}" if is_min_p else f"TopK Temperature: {temperature}"
    plt.plot(sub_df["accuracy"], sub_df["avg_correct_entropy"], linestyle=linestyle, c=color)
    # Scatter the points, using the min_p or top_p values to choose the color
    if is_min_p:
        colors = [f"C{unique_min_p_values.tolist().index(min_p)}" for min_p in sub_df["options.min_p"]]
        plt.scatter(sub_df["accuracy"], sub_df["avg_correct_entropy"], c=colors)
    else:
        colors = [f"C{unique_top_p_values.tolist().index(top_p)}" for top_p in sub_df["options.top_p"]]
        plt.scatter(sub_df["accuracy"], sub_df["avg_correct_entropy"], c=colors)
for temperature_index, temperature in enumerate(unique_temperatures):
    plt.plot([], linestyle="-", label=f"Temperature: {temperature}", color=f"C{temperature_index}")
for min_p_index, min_p in enumerate(unique_min_p_values):
    plt.scatter([], [], label=f"MinP: {min_p:.1f}", color=f"C{min_p_index}")
for top_p_index, top_p in enumerate(unique_top_p_values):
    plt.scatter([], [], label=f"TopP: {top_p:.1f}", color=f"C{top_p_index}")
plt.xlabel("Accuracy")
plt.ylabel("Average Entropy")
plt.legend()
plt.xlim(0.35,0.50)
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8))

unique_temperatures = df["options.temperature"].unique()
temperature_norm = plt.Normalize(unique_temperatures.min(), unique_temperatures.max())
color_map = plt.cm.viridis

for is_min_p, group_df in df.groupby("is_min_p"):
    marker = 'o' if is_min_p else '^'
    label = 'Min-p' if is_min_p else 'Top-p'
    
    scatter = ax.scatter(group_df["accuracy"], group_df["avg_correct_entropy"],
                         c=group_df["options.temperature"], cmap=color_map, norm=temperature_norm,
                         marker=marker, s=100, alpha=0.7, label=label)
    
    for _, row in group_df.iterrows():
        ax.annotate(f"{row['options.min_p' if is_min_p else 'options.top_p']:.2f}", 
                    (row["accuracy"], row["avg_correct_entropy"]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel("Accuracy")
ax.set_ylabel("Average Entropy of Correct Predictions (\"Creativity\")")
ax.set_title("Comparison of Min-p and Top-p: Accuracy vs Diversity")
ax.legend()

cbar = fig.colorbar(scatter)
cbar.set_label("Temperature")

plt.tight_layout()
plt.show()
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
    ax.plot(points[:, 0], points[:, 1])
    ax.plot([], c=color, linestyle='--', label=label, marker=marker)

for is_min_p, group_df in df.groupby("is_min_p"):
    marker = 'o' if is_min_p else '^'
    label = 'Min-p' if is_min_p else 'Top-p'
    color = 'C0' if is_min_p else 'C1'
    
    plot_sorted_by_diversity(group_df, color, label, marker)
    
    scatter = ax.scatter(group_df["accuracy"], group_df["avg_correct_entropy"],
                         c=group_df["options.temperature"], cmap=color_map, norm=temperature_norm,
                         marker=marker, s=100, alpha=1.0, zorder=10)
    
    # for _, row in group_df.iterrows():
    #     angle = np.random.uniform(0, 2 * np.pi)
    #     offset_x = 10 * np.cos(angle)
    #     offset_y = 10 * np.sin(angle)
    #     ax.annotate(f"{row['options.min_p' if is_min_p else 'options.top_p']:.2f}", 
    #                 (row["accuracy"], row["avg_correct_entropy"]),
    #                 xytext=(offset_x, offset_y), textcoords='offset points', fontsize=8, zorder=20)
    

ax.set_xlabel("Accuracy")
ax.set_ylabel("Avg. Entropy of Correct Predictions (\"Creativity\")")
ax.set_title("Comparison of Min-p and Top-p: Accuracy vs Creativity")
ax.legend()

cbar = fig.colorbar(scatter)
cbar.set_label("Temperature")

plt.tight_layout()
plt.show()
# %%
