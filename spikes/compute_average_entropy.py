# %%
import blackhc.project.script  # noqa F401
import pandas as pd
import wandb
from tqdm.auto import tqdm

# %%
# Setup a wandb run that will transform data.
# This will preserve the data graph and allow us to track the data transformation process
run = wandb.init(project="gsm8k", job_type="diversity_computation", reinit=True)

# %% Load blackhc/gsm8k/gsm8k_cot_self_consistency_eval_samples_exploded:v0
artifact = run.use_artifact(
    "blackhc/gsm8k/gsm8k_cot_self_consistency_eval_samples_exploded:v1"
)

# Download the artifact
artifact_dir = artifact.download()

# %%
# Load
# gsm8k_cot_self_consistency_eval_samples_exploded.feather
# gsm8k_cot_self_consistency_eval_samples_resps_embeddings.feather
samples_exploded = pd.read_feather(
    artifact_dir + "/gsm8k_cot_self_consistency_eval_samples_exploded.feather"
)
samples_resps_embeddings = pd.read_feather(
    artifact_dir + "/gsm8k_cot_self_consistency_eval_samples_resps_embeddings.feather"
)

# %% Extract all the option.* columns
options = samples_exploded[
    [
        col
        for col in samples_exploded.columns
        if col.startswith("options.")
        and not col in ["options.until", "options.do_sample"]
    ]
].drop_duplicates()
options

# %%
# Group samples_exploded by exact_match
samples_exploded_grouped_exact_match = samples_exploded.groupby("exact_match")

# %% Get the group with 1 as df
exact_matches = samples_exploded_grouped_exact_match.get_group(1)


# %% Group these by doc_id
exact_matches_grouped_doc_id = exact_matches.groupby("doc_id")
wrong_matches_grouped_doc_id = samples_exploded_grouped_exact_match.get_group(
    0
).groupby("doc_id")

# %% Iterate over the groups
import numpy as np

THRESHOLD = 1e-6

def compute_entropy(evs, threshold):
    clamped_evs = evs[evs > threshold]
    entropy = 0.5 * (
        np.sum(np.log(clamped_evs)) + 0.5 * len(evs) * np.log(2 * np.pi * np.e)
    )
    return entropy


def diversity_score(evs, threshold=THRESHOLD):
    clamped_evs = evs[evs > threshold]
    entropy_part = 0.5 * np.sum(np.log(clamped_evs))
    noise_entropy = 0.5 * (
        np.log(threshold) * len(clamped_evs)
    )
    return entropy_part - noise_entropy


def compute_diversity_score_for_group(group_by_doc_id):
    doc_id_entropies = {}
    for doc_id, group in tqdm(group_by_doc_id):
        # Skip if group.resps is all empty
        if not all(resp != "" for resp in group.resps):
            continue
        # Look up group.resps in samples_resps_embeddings' index
        resps_embeddings = samples_resps_embeddings.loc[group.resps]
        # Convert to numpy array
        resps_embeddings_numpy = resps_embeddings.to_numpy()
        # Center by subtracting the mean
        resps_embeddings_numpy -= np.mean(resps_embeddings_numpy, axis=0)
        # Compute the EVs
        evs = np.linalg.eigvalsh(resps_embeddings_numpy @ resps_embeddings_numpy.T)
        entropy = diversity_score(evs)
        # print(f"Entropy for doc_id {doc_id}: {entropy}")
        doc_id_entropies[doc_id] = entropy
    return doc_id_entropies


doc_id_entropies_exact_match = compute_diversity_score_for_group(exact_matches_grouped_doc_id)

doc_id_entropies_wrong_match = compute_diversity_score_for_group(wrong_matches_grouped_doc_id)

# %%
# Plot the sorted entropies
import matplotlib.pyplot as plt

doc_id_entropies_exact_match_sorted = sorted(doc_id_entropies_exact_match.values())
doc_id_entropies_wrong_match_sorted = sorted(doc_id_entropies_wrong_match.values())

x = np.arange(len(doc_id_entropies_exact_match_sorted)) / len(
    doc_id_entropies_exact_match_sorted
)
plt.plot(x, doc_id_entropies_exact_match_sorted, label="Correct")
x = np.arange(len(doc_id_entropies_wrong_match_sorted)) / len(
    doc_id_entropies_wrong_match_sorted
)
plt.plot(x, doc_id_entropies_wrong_match_sorted, label="Wrong")
plt.legend()
plt.xlabel("Answer")
plt.ylabel("Entropy")
plt.title(f"Entropy of Responses for Each Question\n({options.iloc[0].to_dict()})")
plt.show()

# %%
