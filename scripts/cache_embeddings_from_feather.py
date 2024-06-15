#%%
import blackhc.project.script
import embedding_cache


#%% Load spikes/gsm8k_cot_self_consistency_eval_samples_resps_embeddings.feather
import pandas as pd

df = pd.read_feather(f'spikes/gsm8k_cot_self_consistency_eval_samples_resps_embeddings.feather')

#%%
cache = embedding_cache.EmbeddingCache()

#%%
for index, row in df.iterrows():
    print(index)
    print(row.to_numpy())
    break

#%%
from tqdm.auto import tqdm

for index, row in tqdm(df.iterrows(), total=len(df)):
    cache.put(index, row.to_numpy())
#%%