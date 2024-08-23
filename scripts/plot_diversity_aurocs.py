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
df["error"] = 1.0 - df["accuracy"]
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


# def plot_sorted_by_diversity(data, color, label, marker):
#     points = data[["accuracy", "avg_correct_entropy"]].values
#     points = np.array(points)
#     points = points[points[:, 1].argsort()]
#     ax.plot(points[:, 0], points[:, 1], c=color)
#     ax.plot([], c=color, linestyle='-', label=label, marker=marker)

from scipy.spatial import ConvexHull

def plot_data_and_pareto_frontier(data, color, label, marker):
    points = data[["accuracy", "avg_correct_entropy"]].values.astype(float)
    # Drop rows with NaNs in them
    points = points[~np.isnan(points).any(axis=1)]
       
    # Plot the data points
    # ax.scatter(points[:, 0], points[:, 1], c=color, marker=marker, label=label)
    
    # Compute and plot the convex hull
    if len(points) > 2:  # ConvexHull requires at least 3 points
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], c=color)
    
for is_min_p, group_df in reversed(list(df.groupby("is_min_p"))):
    marker = 'o' if is_min_p else '^'
    label = 'Min-p' if is_min_p else 'Top-p'
    color = 'C0' if is_min_p else 'C1'
    level = group_df["options.min_p"] if is_min_p else group_df["options.top_p"]
    
    plot_data_and_pareto_frontier(group_df, color, label, marker)
    ax.scatter([], [], c=color, marker=marker, label=label)
    
    scatter = ax.scatter(group_df["accuracy"], group_df["avg_correct_entropy"],
                         c=group_df["options.temperature"], cmap=color_map, norm=temperature_norm,
                         marker=marker, s=100*level/0.6, alpha=1.0, zorder=10)
     
ax.set_xlabel("Accuracy")
ax.set_ylabel("Avg. Entropy of Correct Predictions (\"Creativity\")")
ax.set_title("Comparison of Min-p and Top-p: Accuracy vs Creativity")
ax.set_xlim(0.38,0.50)
ax.legend(frameon=True)
# Change the x-axis label to be in percentage
import matplotlib.ticker as mtick
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))


if True:
    # Create a smaller subplot within the first subplot
    # Manually transform from axes coordinates to figure coordinates
    inset_position = ax.transAxes.transform((0.05, 0.45))
    inset_position = fig.transFigure.inverted().transform(inset_position)
    inset_ax = fig.add_axes([inset_position[0], inset_position[1], 0.25, 0.25])  # [left, bottom, width, height]

    # Scatter all points in the smaller subplot
    for is_min_p, group_df in df.groupby("is_min_p"):
        marker = 'o' if is_min_p else '^'
        color = 'C0' if is_min_p else 'C1'
        level = group_df["options.min_p"] if is_min_p else group_df["options.top_p"]
    
        inset_ax.scatter(group_df["accuracy"], group_df["avg_correct_entropy"],
                        c=color,
                        marker=marker, s=20*level/0.6, alpha=0.7, zorder=10)

    inset_ax.set_xlabel("Accuracy", fontsize=8)
    inset_ax.set_ylabel("Avg. Entropy", fontsize=8)
    inset_ax.tick_params(axis='both', which='major', labelsize=8)

    # Add a frame around the inset
    inset_ax.spines['top'].set_visible(True)
    inset_ax.spines['right'].set_visible(True)
    inset_ax.spines['bottom'].set_visible(True)
    inset_ax.spines['left'].set_visible(True)

    # Make the frame thicker and add a light gray background
    for spine in inset_ax.spines.values():
        spine.set_linewidth(1.5)
    # inset_ax.set_facecolor('#f0f0f0')

    # Adjust the position of x and y axis labels
    inset_ax.xaxis.set_label_coords(0.5, 0.15)
    inset_ax.yaxis.set_label_coords(0.15, 0.5)
    inset_ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
    # Disable the grid for inset_ax
    inset_ax.xaxis.set_major_locator(mtick.MultipleLocator(0.2))
    inset_ax.xaxis.set_minor_locator(mtick.MultipleLocator(0.05))
    inset_ax.yaxis.set_major_locator(mtick.MultipleLocator(100))
    inset_ax.yaxis.set_minor_locator(mtick.MultipleLocator(50))
    inset_ax.grid(True, which='minor', linestyle='--', linewidth=0.5)

cbar = fig.colorbar(scatter)
cbar.set_label("Temperature")

plt.tight_layout()
# Save as pdf
plt.savefig("diversity_aurocs.pdf", bbox_inches='tight')
plt.show()
# %%
pivot_table = df.pivot_table(
    values=["accuracy", "avg_correct_entropy"],
    index=["is_min_p", "options.temperature", "options.top_p", "options.min_p"],
    aggfunc=np.mean
)

pivot_table

# %%
subdf = df[df["is_min_p"] == True][["options.temperature", "options.min_p"]]
# Rename options.min_p to min_p
subdf.rename(columns={"options.min_p": "min_p", "options.temperature": "temperature"}, inplace=True)
records = subdf.to_records(index=False)
records.tolist()
# %%
subdf = df[df["is_min_p"] == False][["options.temperature", "options.top_p"]]
# Rename options.top_p to top_p
subdf.rename(columns={"options.top_p": "top_p", "options.temperature": "temperature"}, inplace=True)
records = subdf.to_records(index=False)
records.tolist()
# %%
