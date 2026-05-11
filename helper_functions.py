import numpy as np
import ast


def memory_complexity(complexity, epoch):
    avg_epoch_time = round(complexity['Epoch time(s)'].mean(), 4)
    avg_memory = round(complexity['Memory (MB)'].mean(), 4)

    print(f"AVERAGE EPOCH TIME: {avg_epoch_time}s")
    print(f"TOTAL TRAINING TIME: {avg_epoch_time * epoch}s")
    print(f"AVERAGE MEMORY CONSUMPTION PER EPOCH: {avg_memory} MB")

def calculate_recomm(recomm_file, top_k_):
    user_ground_truths = convert_str_to_list(recomm_file["user_ground_truths"])
    user_top_k_prediction = convert_str_to_list(recomm_file["user_top_k_prediction"])
    performance_measures = dict()

    for top_k in top_k_:
        
        performance_measures["Recall@"+str(top_k)]= Recall(user_ground_truths, user_top_k_prediction, top_k)
        performance_measures["NDCG@"+str(top_k)]= NDCG(user_ground_truths, user_top_k_prediction, top_k)

    for key, value in performance_measures.items():
        print(f"{key:<20} | {value}")


def print_dataset_name(dataset_name="Dataset"):
    # ANSI escape codes for colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    MAGENTA = "\033[95m"
    print(f"{BOLD}{MAGENTA}{dataset_name}{RESET}")
    


def print_dataset_info(train_users, test_users, valid_users, dataset_name="Dataset"):
    # ANSI escape codes for colors
    RESET = "\033[0m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    
    print(f"{CYAN}Number of users in training data:{RESET} {GREEN}{train_users}{RESET}")
    if valid_users > 0:
        print(f"{CYAN}Number of users in validation data:{RESET} {GREEN}{valid_users}{RESET}")
    print(f"{CYAN}Number of users in test data:{RESET} {YELLOW}{test_users}{RESET}")    


def formating1(model_name):

    line = "=" * 45
    print(line)
    print(f"{model_name.upper():^45}")

def formating2(dataset_name):
    print(f"Dataset: {dataset_name.upper()}")
    

    

def convert_str_to_list(string_to_list):
    string_to_list = [ast.literal_eval(i) for i in string_to_list]
    return string_to_list


def Recall(ground_turth, predictions, top_k):

    res = []
    for idx in range(len(ground_turth)):
        gt = ground_turth[idx]
        pred = predictions[idx][:top_k]
        rec = np.in1d(pred, list(gt)).sum() / len(gt)
        res.append(rec)

    return round(np.mean(res), 4)

def HR(ground_turth, predictions, top_k):
    res = []

    for idx in range(len(ground_turth)):
        gt = ground_turth[idx]
        pred = predictions[idx][:top_k]
        r = np.in1d(pred, list(gt))
        res.append(1 if r.sum() else 0)
    return round(np.mean(res), 4)


import numpy as np

def NDCG(ground_truth, predictions, top_k):
    def DCG(rel):
        rel = np.asarray(rel, dtype=np.float32)
        if rel.size == 0:
            return 0.0
        return np.sum((2**rel - 1) / np.log2(np.arange(2, rel.size + 2)))

    res = []

    for gt, pred in zip(ground_truth, predictions):
        pred = pred[:top_k]
        gt_set = set(gt)

        # binary relevance for the ranked predictions
        r = np.isin(pred, list(gt_set)).astype(np.float32)

        # correct ideal relevance vector
        ideal_len = min(len(gt_set), top_k)
        ideal_r = np.zeros(top_k, dtype=np.float32)
        ideal_r[:ideal_len] = 1.0

        idcg = DCG(ideal_r)
        ndcg = 0.0 if idcg == 0 else DCG(r) / idcg
        res.append(ndcg)

    return round(float(np.mean(res)) if res else 0.0, 4)
