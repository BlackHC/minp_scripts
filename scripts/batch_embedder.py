#%% 
import blackhc.project.script # noqa F401
import pandas as pd
import wandb
from wandb.apis.public.runs import Run
from tqdm.auto import tqdm
import tiktoken
import openai
import embedding_cache

#%%
cache = embedding_cache.EmbeddingCache()
client = openai.Client()
model = "text-embedding-3-small"

#%%
# %%
api = wandb.Api()

#%%
# Load all runs from project "lm-eval-harness-integration"
runs = api.runs("menhguin/lm-eval-harness-integration")
print(f"Found {len(runs)} runs")
# %%
run: Run
for run in tqdm(list(runs)[::-1], desc="Processing runs"):
    if run.state != "finished":
        continue
    if run.jobType is None:
        print(run.name)
        if "task_configs" not in run.config:
            print("No task configs found. Continuing.")
            continue
        tasks = list(run.config["task_configs"].keys())
        assert len(tasks) == 1, f"Expected exactly one task, but found {tasks}"
        task = tasks[0]
        match task:
            case "gsm8k_cot_self_consistency":
                repeats = run.config["task_configs"][task]["repeats"]
                task_desc = f"GSM8k,COT,SC,maj@{repeats}"
            case _:
                print("Unsupported task. Continuing.")
                continue
        top_p = run.config["cli_configs"]["gen_kwargs"]["top_p"]
        min_p = run.config["cli_configs"]["gen_kwargs"].get("min_p", None)
        temperature = run.config["cli_configs"]["gen_kwargs"]["temperature"]
    else:
        print(f"Run {run.name} is certainly not a generation run. Skipping.")
        continue
        
    # Processing finished run
    # print(f"Processing run {run.name}...")
    # Check if the job already has a embedding_computation run
    logged_artifacts = list(run.logged_artifacts())
    already_processed = False
    gsm8k_cot_self_consistency_artifact = None
    for logged_artifact in logged_artifacts:
        # print(logged_artifact.name)
        if logged_artifact.name.startswith("gsm8k_cot_self_consistency:"):
            gsm8k_cot_self_consistency_artifact = logged_artifact
        if not already_processed:
            for used_by_run in logged_artifact.used_by():
                if used_by_run.jobType == "embedding_computation" and used_by_run.state == "finished":
                    already_processed = True
                    break
    if already_processed:
        print(f"Embedding computation run for {run.name} already exists. Skipping.")
        continue
    if gsm8k_cot_self_consistency_artifact is None:
        print(f"No gsm8k_cot_self_consistency artifact found for run {run.name}. Skipping.")
        continue
    print(f"⏳ Found gsm8k_cot_self_consistency artifact for run {run.name}. Embedding.")
    print()
    print("📖 Temperature:", temperature)
    print("📖 Top-p:", top_p)
    print("📖 Min-p:", min_p)
    print()
    processor_run = wandb.init(entity="menhguin", project='lm-eval-harness-integration', job_type='embedding_computation', reinit=True, name=f"Embedder: {run.name}",
                               config={"original_run": run.id, "original_config": run.config, "artiface_name": gsm8k_cot_self_consistency_artifact.name})
    processor_run.use_artifact(gsm8k_cot_self_consistency_artifact)
    # Download the artifact
    artifact_dir = gsm8k_cot_self_consistency_artifact.download()

    # Load gsm8k_cot_self_consistency_eval_samples_other.json from the artifact
    json_df = pd.read_json(artifact_dir + '/gsm8k_cot_self_consistency_eval_samples.json')
    # The doc column is a dict. Let's expand it into columns
    df = pd.concat([json_df.drop(['doc'], axis=1), json_df['doc'].apply(pd.Series)], axis=1)
    # The resps column is a list of lists. Let's explode it into rows
    df = df.explode('resps')
    df = df.explode('resps')

    # Get all unique resps
    unique_resps = df['resps'].unique()
    # Drop empty resps
    unique_resps = [resp for resp in unique_resps if resp]
    len(unique_resps)

    # Count tiktoken in each resp and sum them
    encoder = tiktoken.encoding_for_model("gpt-4")

    all_tokens = encoder.encode_ordinary_batch(list(unique_resps))
    # Sum the tiktoken counts
    num_tokens = sum([len(tokens) for tokens in all_tokens])
    print(num_tokens)

    # arguments is a list of lists. Let's explode it once and then we take the first argument as "examples"
    df = df.explode('arguments')

    # Assuming 'arguments' is a list of lists with exactly two elements each
    df[['examples', 'options']] = pd.DataFrame(df['arguments'].tolist(), index=df.index)
    # Options is a dict. Let's expand it into columns with 'options.' as prefix
    df = pd.concat([df.drop(['options'], axis=1), df['options'].apply(lambda x: pd.Series(x)).add_prefix('options.')], axis=1)

    # Drop arguments
    df.drop('arguments', axis=1, inplace=True)

    # Cast arguments to str
    # df['examples'] = df['examples'].astype(str)

    # Drop filtered_resps
    df.drop('filtered_resps', axis=1, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    # Save the df to a feather file
    df.to_feather('gsm8k_cot_self_consistency_eval_samples_exploded.feather')
    
    # Look up embeddings in the cache
    unique_embeddings = list(cache.get_batch(unique_resps))
    # Create a dict to map resps to embeddings
    resp_embedding_dict = dict(zip(unique_resps, unique_embeddings))
    
    # Find all the embeddings that are None rn
    embeddings_to_compute = [k for k, v in resp_embedding_dict.items() if v is None]
    
    # Embed the resps using the model text-embedding-3-small
    # Batch processing of unique_resps
    batch_size = 2048  # Define the batch size
    embedding_results = []
    for i in tqdm(range(0, len(embeddings_to_compute), batch_size)):
        batch = list(embeddings_to_compute[i:i+batch_size])
        # Check if any batch elements are empty
        assert all(batch), f"Batch {i//batch_size} contains empty elements"
        result = client.embeddings.create(input=batch, model=model)
        embedding_results.extend(result.data)

    computed_embeddings = [item.embedding for item in embedding_results]
    cache.put_batch(embeddings_to_compute, computed_embeddings)
    # Update resp_embedding_dict
    resp_embedding_dict.update(dict(zip(embeddings_to_compute, computed_embeddings)))
    # Turn this into a dataframe
    resp_embedding_df = pd.DataFrame.from_dict(resp_embedding_dict, orient='index')
    # Save the embedding df to another feather file
    resp_embedding_df.to_feather('gsm8k_cot_self_consistency_eval_samples_resps_embeddings.feather')
    # Create a new artifact to store the transformed data
    artifact = wandb.Artifact('gsm8k_cot_self_consistency_eval_samples_exploded', type='data')
    artifact.add_file('gsm8k_cot_self_consistency_eval_samples_exploded.feather')
    artifact.add_file('gsm8k_cot_self_consistency_eval_samples_resps_embeddings.feather')
    # Log the artifact
    processor_run.log_artifact(artifact)
    # Finish wandb
    processor_run.finish()
    print(f"✅ Finished processing run {run.name}")
    print()
    print()
# %%
