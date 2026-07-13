import pandas as pd
from pathlib import Path
from helper_functions import *


def change_data_elliot_format(_dict):
    _df = (
        pd.Series(_dict, name="ItemID")
        .explode()
        .reset_index()
        .rename(columns={"index": "UserID"})
    )
    # optional: ensure integers (in case of missing/strings)
    _df["UserID"] = _df["UserID"].astype(int)
    _df["ItemID"] = _df["ItemID"].astype(int)
    _df["Rating "] = 1

    return _df


def read_data(path, split_name):
    data_dict = {}

    with open(path, "r") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                print(f"Empty line in {split_name} at line {line_num}")
                continue

            try:
                parts = line.split()   # important: handles multiple spaces/tabs safely

                uid = int(parts[0])
                items = [int(i) for i in parts[1:]]

                if len(items) == 0:
                    print(f"User {uid} has no items in {split_name} at line {line_num}")
                    continue

                data_dict[uid] = items

            except Exception as e:
                print(f"Bad line in {split_name} at line {line_num}")
                print(f"Raw line: {repr(raw_line)}")
                print(f"Parsed parts: {parts if 'parts' in locals() else None}")
                print(f"Error: {type(e).__name__}: {e}")
                continue

    return data_dict

def training_data(path):
    train_dict = read_data(path, "training data")

    new_train_dict = dict()
    valid_dict = dict()

    for key, items_list in train_dict.items():

        if len(items_list) > 1:
            valid_dict[key] = [ items_list[-1] ]
            new_train_dict[key] = items_list[:-1]
        else:
            new_train_dict[key] = items_list

    
    train_df = change_data_elliot_format(new_train_dict)
    valid_df = change_data_elliot_format(valid_dict)
    return train_df, valid_df

def testing_data(path):
    test_dict = read_data(path, "test data")
    test_df = change_data_elliot_format(test_dict)
    return test_df


print("\n" + "★"*44)
print("PREPARE DATA FOR NGCF MODEL".center(44))
print("★"*44 + "\n")

###### GOWALLA NGCF
print_dataset_name(dataset_name="GOWALLA")
path = "ngcf_data/gowalla"
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/ngcf/tuning/gowalla/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
valid_df.to_csv(path / "valid.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)

train_users = len(train_df.UserID.unique())
valid_users = len(valid_df.UserID.unique())
test_users = len(test_df.UserID.unique())

print_dataset_info( train_users , test_users, valid_users,    dataset_name = "GOWALLA")

######## AMAZON-BOOK NGCF
path = "ngcf_data/amazon-book"
print_dataset_name(dataset_name="AMAZON-BOOK")
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/ngcf/tuning/amazon-book/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
valid_df.to_csv(path / "valid.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)

train_users = len(train_df.UserID.unique())
valid_users = len(valid_df.UserID.unique())
test_users = len(test_df.UserID.unique())
print_dataset_info( train_users , test_users, valid_users,   dataset_name = "AMAZON-BOOK")


print("\n" + "★"*44)
print("PREPARE DATA FOR LIGHTGCN MODEL".center(44))
print("★"*44 + "\n")


######### LIGHTGCN GOWALLA
path = "lightgcn_data/gowalla"
print_dataset_name(dataset_name="GOWALLA")
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/lightgcn/tuning/gowalla/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
valid_df.to_csv(path / "valid.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)

train_users = len(train_df.UserID.unique())
valid_users = len(valid_df.UserID.unique())
test_users = len(test_df.UserID.unique())
print_dataset_info( train_users , test_users, valid_users,  dataset_name = "GOWALLA")


########## LIGHTGCN AMAZON-BOOK
path = "lightgcn_data/amazon-book"
print_dataset_name(dataset_name="AMAZON-BOOK")
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/lightgcn/tuning/amazon-book/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
valid_df.to_csv(path / "valid.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)

train_users = len(train_df.UserID.unique())
valid_users = len(valid_df.UserID.unique())
test_users = len(test_df.UserID.unique())
print_dataset_info( train_users , test_users, valid_users,  dataset_name = "AMAZON-BOOK")


############ LIGHTGCN YELP2018 
path = "lightgcn_data/yelp2018"
print_dataset_name(dataset_name="YELP2018")
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/lightgcn/tuning/yelp2018/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
valid_df.to_csv(path / "valid.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)

train_users = len(train_df.UserID.unique())
valid_users = len(valid_df.UserID.unique())
test_users = len(test_df.UserID.unique())
print_dataset_info( train_users , test_users, valid_users,   dataset_name = "YELP2018")



print(r"""
  _____   ____   _   _  ______
 |  __ \ / __ \ | \ | ||  ____|
 | |  | | |  | ||  \| || |__
 | |  | | |  | || . ` ||  __|
 | |__| | |__| || |\  || |____
 |_____/ \____/ |_| \_||______|
""")
