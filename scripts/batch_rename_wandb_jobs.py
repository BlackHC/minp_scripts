#%%
import blackhc.project.script
#%%
import wandb
from wandb.apis.public.runs import Run
from tqdm.auto import tqdm

# %%
api = wandb.Api()

#%%
# Load all runs from project "lm-eval-harness-integration"
runs = api.runs("blackhc/lm-eval-harness-integration")
print(f"Found {len(runs)} runs")
run_list = [run for run in tqdm(runs)]
# %%
run: Run
for run in runs:
    if run.state != "finished":
        continue
    if run.jobType is None:
        print(run.id) 
        top_p = run.config["cli_configs"]["gen_kwargs"]["top_p"]
        min_p = run.config["cli_configs"]["gen_kwargs"].get("min_p", None)
        temperature = run.config["cli_configs"]["gen_kwargs"]["temperature"]
        tasks = list(run.config["task_configs"].keys())
        assert len(tasks) == 1, f"Expected exactly one task, but found {tasks}"
        task = tasks[0]
        match task:
            case "gsm8k_cot_self_consistency":
                repeats = run.config["task_configs"][task]["repeats"]
                task_desc = f"GSM8k,COT,SC,maj@{repeats}"
            case _:
                print("Unsupported task")
                continue
        new_name = f"{task_desc},top_p={top_p},min_p={min_p},temp={temperature},{run.id}"
        print(new_name)
        run.name = new_name
        run.save()
# %%
