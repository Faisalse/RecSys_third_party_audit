from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function
import yaml
import os
import time
import pandas as pd
from pathlib import Path

tuning_start = time.time()


hp = HyperTuning(objective_function=objective_function, algo='bayes', early_stop=10,
                max_evals=30, params_file='ngcf.hyper', fixed_config_file_list=['fixed_dataset_for_ngcf_tuning.yaml'])

with open(hp.fixed_config_file_list[0] , "r") as file:
    config = yaml.safe_load(file)
output_dir = Path('results/'+config['model']+"/"+config["dataset"])
os.makedirs(output_dir, exist_ok=True)
output_path_file = os.path.join(output_dir, "hyper_result.txt")


hp.run()
total_tuning_time = time.time() - tuning_start
tuning_time_memory_usuage = pd.DataFrame()
tuning_time_memory_usuage["tuning_time (s)"] = [total_tuning_time]


hp.export_result(output_file = output_path_file)
print("best params: ", hp.best_params)
print("best result: ")
print(hp.params2result[hp.params2str(hp.best_params)])

tuning_time_memory_usuage.to_csv(
    output_dir / "tuning_time_memory_usuage.txt",
    index=False,
    sep="\t"
)