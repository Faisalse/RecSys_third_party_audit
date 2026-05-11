from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function
import yaml
import os
import time
import threading
import pynvml
import pandas as pd


tuning_start = time.time()

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

max_gpu_util = 0
max_mem_used = 0
running = True

def monitor_gpu():
    global max_gpu_util, max_mem_used, running

    while running:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        max_gpu_util = max(max_gpu_util, util.gpu)
        max_mem_used = max(max_mem_used, mem.used)

        time.sleep(0.5)


hp = HyperTuning(objective_function=objective_function, algo='bayes', early_stop=10,
                max_evals=5, params_file='ngcf.hyper', fixed_config_file_list=['fixed_dataset_for_lightgcn_tuning.yaml'])

with open(hp.fixed_config_file_list[0] , "r") as file:
    config = yaml.safe_load(file)
output_dir = 'results/'+config['model']+"/"+config["dataset"]
os.makedirs(output_dir, exist_ok=True)
output_path_file = os.path.join(output_dir, "hyper_result.txt")


total_tuning_time = time.time() - tuning_start
monitor_thread = threading.Thread(target=monitor_gpu)
monitor_thread.start()

########## RUN TUNING
#breakpoint()
hp.run()


running = False
monitor_thread.join()
print(f"Maximum GPU Utilization: {max_gpu_util}%")
print(f"Maximum GPU Memory Used: {max_mem_used / 1024**3:.2f} GB")
pynvml.nvmlShutdown()


tuning_time_memory_usuage = pd.DataFrame()
tuning_time_memory_usuage["tuning_time (s)"] = [total_tuning_time]
tuning_time_memory_usuage["Maximum GPU Utilization"] = [max_gpu_util] 
tuning_time_memory_usuage["Maximum GPU Memory Used (GB)"] = [max_mem_used / 1024**3] 




hp.export_result(output_file = output_path_file)
print("best params: ", hp.best_params)
print("best result: ")
print(hp.params2result[hp.params2str(hp.best_params)])

tuning_time_memory_usuage.to_csv(
    output_dir / "tuning_time_memory_usuage.txt",
    index=False,
    sep="\t"
)