import argparse
from pathlib import Path
import pandas as pd
from helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="daisyrec", help="original_ngcf, " \
"original_lightgcn, daisyrec, recbole, elliot")
parser.add_argument("--top_k", type=int, nargs="+", default=[20])
args = parser.parse_args()

# user_ground_truths
# user_top_k_prediction"

if args.model_name == "original_ngcf":

    ### GOWALLA
    model_name = "original_ngcf"
    formating1(model_name)
    path = "results/orginal_ngcf/gowalla/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("GOWALLA")
    calculate_recomm(recomm_file, args.top_k)
    
    ### AMAZON-BOOK
    path = "results/orginal_ngcf/amazon-book/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("AMAZON BOOK")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)


elif args.model_name == "original_lightgcn":

    ### GOWALA
    model_name = "original_lightgcn"
    formating1(model_name)
    path = "results/original_lightgcn/gowalla/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("GOWALLA")
    calculate_recomm(recomm_file, args.top_k)
    
    ### YELP 2018
    path = "results/original_lightgcn/yelp2018/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("YELP 2018")
    calculate_recomm(recomm_file, args.top_k)
    
    ### AMAZON-BOOK
    path = "results/original_lightgcn/amazon-book/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("AMAZON BOOK")
    calculate_recomm(recomm_file, args.top_k)

    line = "=" * 45
    print(line)


elif args.model_name == "daisyrec":
    
    model_name = "daisyrec_ngcf"
    formating1(model_name)
    
    # GOWALA_SAMPLE SIZE 100
    path = "results/daisyrec/ngcf/with_reported_HP/gowalla/sampling_100/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("GOWALLA_SAMPLE SIZE 100")
    calculate_recomm(recomm_file, args.top_k)

    # GOWALA_SAMPLE SIZE 500
    path = "results/daisyrec/ngcf/with_reported_HP/gowalla/sampling_500/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("GOWALLA_SAMPLE SIZE 500")
    calculate_recomm(recomm_file, args.top_k)

    # GOWALA_SAMPLE SIZE 1000
    path = "results/daisyrec/ngcf/with_reported_HP/gowalla/sampling_1000/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("GOWALLA_SAMPLE SIZE 1000")
    calculate_recomm(recomm_file, args.top_k)

    # GOWALA_SAMPLE SIZE 40981
    path = "results/daisyrec/ngcf/with_reported_HP/gowalla/sampling_40981/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("GOWALLA FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)

    print("GOWALLA Hyperparameter tuning results")
    path = "results/daisyrec/ngcf/with_hyperparameter_tuning/gowalla/sampling_30000/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("GOWALLA SAMPLE SIZE 30000")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)


    ######################################################################
    # AMAZON-BOOK_SAMPLE SIZE 100
    path = "results/daisyrec/ngcf/with_reported_HP/amazon-book/sampling_100/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("AMAZON-BOOK SAMPLE SIZE 100")
    calculate_recomm(recomm_file, args.top_k)
    
    # AMAZON-BOOK_ SIZE 500
    path = "results/daisyrec/ngcf/with_reported_HP/amazon-book/sampling_500/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("AMAZON-BOOK SAMPLE SIZE 500")
    calculate_recomm(recomm_file, args.top_k)

    # AMAZON-BOOK_ SIZE 1000
    path = "results/daisyrec/ngcf/with_reported_HP/amazon-book/sampling_1000/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("AMAZON-BOOK SAMPLE SIZE 1000")
    calculate_recomm(recomm_file, args.top_k)
    

    # AMAZON-BOOK_ SIZE FULL evaluation
    path = "results/daisyrec/ngcf/with_reported_HP/amazon-book/sampling_91599/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("AMAZON-BOOK SAMPLE SIZE 91599")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)
    

    

    """
    # LIGHTGCN MODEL
    model_name = "daisyrec_lightgcn"
    formating1(model_name)
    # GOWALLA
    path = "results/daisyrec/lightgcn/gowalla/sampling_100/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("GOWALLA SAMPLE SIZE 100")
    calculate_recomm(recomm_file, args.top_k)

    path = "results/daisyrec/lightgcn/gowalla/sampling_500/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("GOWALLA SAMPLE SIZE 500")
    calculate_recomm(recomm_file, args.top_k)
    
    path = "results/daisyrec/lightgcn/gowalla/sampling_1000/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("GOWALLA SAMPLE SIZE 1000")
    calculate_recomm(recomm_file, args.top_k)

    path = "results/daisyrec/lightgcn/gowalla/sampling_40981/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("GOWALLA FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)

    # YELP2018
    path = "results/daisyrec/lightgcn/yelp2018/sampling_100/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("YELP2018 SAMPLE SIZE 100")
    calculate_recomm(recomm_file, args.top_k)

    path = "results/daisyrec/lightgcn/yelp2018/sampling_500/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("YELP2018 SAMPLE SIZE 500")
    calculate_recomm(recomm_file, args.top_k)

    path = "results/daisyrec/lightgcn/yelp2018/sampling_1000/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("YELP2018 SAMPLE SIZE 1000")
    calculate_recomm(recomm_file, args.top_k)

    path = "results/daisyrec/lightgcn/yelp2018/sampling_38048/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "prediction_files.txt", sep = "\t")
    recomm_file["user_ground_truths"] = recomm_file["user_ground_turth"]
    recomm_file["user_top_k_prediction"] = recomm_file["predictions"]
    formating2("YELP2018 FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)
    """

elif args.model_name == "recbole":
    print("^^^^^^^ RESULTS WITH REPORTED HYPERPARAMETERS ^^^^^^^^^^^^^")
    model_name = "recbole_NGCF"
    formating1(model_name)
    # GOWALLA 
    path = "RecBole/results/with_reported_HP/NGCF/gowalla/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("GOWALLA FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)

    # AMAZON-BOOK
    path = "RecBole/results/with_reported_HP/NGCF/amazon-book/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("AMAZON-BOOK FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)
    
    #####################################################################################
    model_name = "recbole_LIGHTGCN"
    formating1(model_name)
    # GOWALLA 
    path = "RecBole/results/with_reported_HP/LightGCN/gowalla/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("GOWALLA FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)
    
    # YELP2018
    path = "RecBole/results/with_reported_HP/LightGCN/yelp2018/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("YELP2018 FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    
    # AMAZON-BOOK
    path = "RecBole/results/with_reported_HP/LightGCN/amazon-book/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("AMAZON-BOOK FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)



    print("^^^^^^^ RESULTS WITH TUNED HYPERPARAMETERS (Bayesian Optimization) ^^^^^^^^^^^^^")
    
    model_name = "recbole_NGCF"
    formating1(model_name)
    # GOWALLA 
    path = "RecBole/results/with_parameter_tuning/NGCF/gowalla/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("GOWALLA FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)

    # AMAZON-BOOK
    path = "RecBole/results/with_parameter_tuning/NGCF/amazon-book/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("AMAZON-BOOK FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)
    
    #####################################################################################
    model_name = "recbole_LIGHTGCN"
    formating1(model_name)
    # GOWALLA 
    path = "RecBole/results/with_parameter_tuning/LightGCN/gowalla/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("GOWALLA FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)
    
    # YELP2018
    path = "RecBole/results/with_parameter_tuning/LightGCN/yelp2018/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("YELP2018 FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45

    # AMAZON-BOOK
    path = "RecBole/results/with_parameter_tuning/LightGCN/amazon-book/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "recommendation_files.csv", sep = "\t")
    formating2("AMAZON-BOOK FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)



elif args.model_name == "elliot":
    from prepare_data_for_elliot_training import *
    
    # GOWALLA
    model_name = "ELLIOT NGCF"
    formating1(model_name)
    path = "elliot/results/with_reported_HP/ngcf/gowalla/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "NGCF_seed=42_e=400_bs=1024_lr=0$0001_factors=64_l_w=1e-05_n_layers=1_weight_size=64_node_dropout=0$1_message_dropout=0$1_it=400.tsv", sep = "\t")
    recomm_file.columns = ["UserID", "ItemID", "score"]
    predictions = recomm_file.groupby("UserID")["ItemID"].apply(list)
    
    path = "elliot/data/ngcf/training/gowalla/"
    path = Path(path)
    turth_df = pd.read_csv(path / "test.tsv", sep = "\t")
    turth_df.columns = ["UserID", "ItemID", "rating"]
    turth_values = turth_df.groupby("UserID")["ItemID"].apply(list)
    rows = []
    for index_ in predictions.index:
        rows.append([index_, str(turth_values[index_]),  str(predictions[index_])  ])
    recomm_file = pd.DataFrame(rows, columns=["UserID", "user_ground_truths", "user_top_k_prediction"])
    formating2("GOWALLA FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    
    
    # AMAZN-BOOK
    path = "elliot/results/with_reported_HP/ngcf/amazon-book/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "NGCF_seed=42_e=200_bs=1024_lr=0$0005_factors=64_l_w=1e-05_n_layers=1_weight_size=64_node_dropout=0$1_message_dropout=0$1_it=200.tsv", sep = "\t")
    recomm_file.columns = ["UserID", "ItemID", "score"]
    predictions = recomm_file.groupby("UserID")["ItemID"].apply(list)

    path = "elliot/data/ngcf/training/amazon-book/"
    path = Path(path)
    turth_df = pd.read_csv(path / "test.tsv", sep = "\t")
    turth_df.columns = ["UserID", "ItemID", "rating"]
    turth_values = turth_df.groupby("UserID")["ItemID"].apply(list)

    rows = []
    for index_ in predictions.index:
        if index_ in turth_values.index and index_ in predictions.index:
            rows.append([index_, str(turth_values[index_]),  str(predictions[index_])  ])
    recomm_file = pd.DataFrame(rows, columns=["UserID", "user_ground_truths", "user_top_k_prediction"])
    formating2("AMAZON-BOOK FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)

    ################################### 
    model_name = "ELLIOT LIGHTGCN"
    formating1(model_name)
    # GOWALLA
    path = "elliot/results/with_reported_HP/lightgcn/gowalla/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "LightGCN_seed=42_e=1000_bs=2048_lr=0$001_factors=64_l_w=0$0001_n_layers=1_it=1000.tsv", sep = "\t", header=None)
    recomm_file.columns = ["UserID", "ItemID", "score"]
    predictions = recomm_file.groupby("UserID")["ItemID"].apply(list)
    
    path = "elliot/data/lightgcn/training/gowalla/"
    path = Path(path)
    turth_df = pd.read_csv(path / "test.tsv", sep = "\t")
    turth_df.columns = ["UserID", "ItemID", "rating"]
    turth_values = turth_df.groupby("UserID")["ItemID"].apply(list)
    rows = []
    
    for index_ in predictions.index:
        rows.append([index_, str(turth_values[index_]),  str(predictions[index_])  ])

    recomm_file = pd.DataFrame(rows, columns=["UserID", "user_ground_truths", "user_top_k_prediction"])
    formating2("GOWALLA FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)

    # AMAZON-BOOK
    path = "elliot/results/with_reported_HP/lightgcn/amazon-book/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "LightGCN_seed=42_e=1000_bs=8192_lr=0$001_factors=64_l_w=0$0001_n_layers=1_it=1000.tsv", sep = "\t", header=None)
    recomm_file.columns = ["UserID", "ItemID", "score"]
    predictions = recomm_file.groupby("UserID")["ItemID"].apply(list)
        
    path = "elliot/data/lightgcn/training/amazon-book/"
    path = Path(path)
    turth_df = pd.read_csv(path / "test.tsv", sep = "\t")
    turth_df.columns = ["UserID", "ItemID", "rating"]
    turth_values = turth_df.groupby("UserID")["ItemID"].apply(list)
    rows = []

    for index_ in predictions.index:
        if index_ in turth_values.index and index_ in predictions.index:
            rows.append([index_, str(turth_values[index_]),  str(predictions[index_])  ])
    recomm_file = pd.DataFrame(rows, columns=["UserID", "user_ground_truths", "user_top_k_prediction"])
    formating2("AMAZON-BOOK FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    

    # yelp2018
    path = "elliot/results/with_reported_HP/lightgcn/yelp2018/"
    path = Path(path)
    recomm_file = pd.read_csv(path / "LightGCN_seed=42_e=1000_bs=2048_lr=0$001_factors=64_l_w=0$0001_n_layers=1_it=1000.tsv", sep = "\t", header=None)
    recomm_file.columns = ["UserID", "ItemID", "score"]
    predictions = recomm_file.groupby("UserID")["ItemID"].apply(list)

    path = "elliot/data/lightgcn/training/yelp2018/"
    path = Path(path)
    turth_df = pd.read_csv(path / "test.tsv", sep = "\t")
    turth_df.columns = ["UserID", "ItemID", "rating"]
    turth_values = turth_df.groupby("UserID")["ItemID"].apply(list)
    rows = []

    for index_ in predictions.index:
        rows.append([index_, str(turth_values[index_]),  str(predictions[index_])  ])

    recomm_file = pd.DataFrame(rows, columns=["UserID", "user_ground_truths", "user_top_k_prediction"])
    formating2("YELP2018 FULL EVALUATION")
    calculate_recomm(recomm_file, args.top_k)
    line = "=" * 45
    print(line)
    
    

    
    
    
    

    
    
    
    














print("+----------------------+")
print("|   COMPLETE RESULTS   |")
print("+----------------------+")

   



    








