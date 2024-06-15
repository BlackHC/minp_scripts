#%% 
import blackhc.project.script # noqa F401
import pandas as pd
import wandb
from tqdm.auto import tqdm

#%%

#%%
# Setup a wandb run that will transform data. 
# This will preserve the data graph and allow us to track the data transformation process
run = wandb.init(project='lm-eval-harness-integration', job_type='embedding_computation', reinit=True)

#%% Load the artifact: "blackhc/lm-eval-harness-integration/gsm8k_cot_self_consistency:v2"
artifact = run.use_artifact('blackhc/lm-eval-harness-integration/gsm8k_cot_self_consistency:v1')

# Download the artifact
artifact_dir = artifact.download()

#%%
# Load gsm8k_cot_self_consistency_eval_samples_other.json from the artifact
json_df = pd.read_json(artifact_dir + '/gsm8k_cot_self_consistency_eval_samples.json')
# %%
# The doc column is a dict. Let's expand it into columns
df = pd.concat([json_df.drop(['doc'], axis=1), json_df['doc'].apply(pd.Series)], axis=1)
# The resps column is a list of lists. Let's explode it into rows
df = df.explode('resps')
df = df.explode('resps')

# %%
# Get all unique resps
unique_resps = df['resps'].unique()
# Drop empty resps
unique_resps = [resp for resp in unique_resps if resp]
len(unique_resps)

# %% Count tiktoken in each resp and sum them
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4")

all_tokens = encoder.encode_ordinary_batch(list(unique_resps))

# %% Sum the tiktoken counts
num_tokens = sum([len(tokens) for tokens in all_tokens])
print(num_tokens)

#%% 
# arguments is a list of lists. Let's explode it once and then we take the first argument as "examples"
df = df.explode('arguments')

# Assuming 'arguments' is a list of lists with exactly two elements each
df[['examples', 'options']] = pd.DataFrame(df['arguments'].tolist(), index=df.index)
# Options is a dict. Let's expand it into columns with 'options.' as prefix
df = pd.concat([df.drop(['options'], axis=1), df['options'].apply(lambda x: pd.Series(x)).add_prefix('options.')], axis=1)

#%% Drop arguments
df.drop('arguments', axis=1, inplace=True)

#%% Cast arguments to str
# df['examples'] = df['examples'].astype(str)

# %% Drop filtered_resps

df.drop('filtered_resps', axis=1, inplace=True)

#%%
df.reset_index(drop=True, inplace=True)
# %% Save the df to a feather file
df.to_feather('gsm8k_cot_self_consistency_eval_samples_exploded.feather')


#%% 
import openai

#%% Embed the resps using the model text-embedding-3-small
model = "text-embedding-3-small"

client = openai.Client()

# Batch processing of unique_resps
batch_size = 2048  # Define the batch size
embedding_results = []
for i in tqdm(range(0, len(unique_resps), batch_size)):
    batch = list(unique_resps[i:i+batch_size])
    # Check if any batch elements are empty
    assert all(batch), f"Batch {i//batch_size} contains empty elements"
    result = client.embeddings.create(input=batch, model=model)
    embedding_results.extend(result.data)

unique_embeddings = [item.embedding for item in embedding_results]

#%% Create a dict to map resps to embeddings
resp_embedding_dict = dict(zip(unique_resps, unique_embeddings))
#%%
# Turn this into a dataframe
resp_embedding_df = pd.DataFrame.from_dict(resp_embedding_dict, orient='index')

# %% Save the embedding df to another feather file
resp_embedding_df.to_feather('gsm8k_cot_self_consistency_eval_samples_resps_embeddings.feather')

# %%
# Create a new artifact to store the transformed data
artifact = wandb.Artifact('gsm8k_cot_self_consistency_eval_samples_exploded', type='data')
artifact.add_file('gsm8k_cot_self_consistency_eval_samples_exploded.feather')
artifact.add_file('gsm8k_cot_self_consistency_eval_samples_resps_embeddings.feather')
# Log the artifact
run.log_artifact(artifact)


# %% Finish wandb
run.finish()
# %%
