import pandas as pd
from pathlib import Path
import os
from helper_functions import *

def training_data(path):
    train_dict = dict()
    try:
        with open(path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    train_dict[uid] = items
    except:
        print(f"A user do not have items in the training data {l}")
    train_df = (
        pd.Series(train_dict, name="ItemID")
        .explode()
        .reset_index()
        .rename(columns={"index": "UserID"})
    )
    # optional: ensure integers (in case of missing/strings)
    train_df["UserID"] = train_df["UserID"].astype(int)
    train_df["ItemID"] = train_df["ItemID"].astype(int)
    train_df["Rating "] = 1
    return train_df

def testing_data(path):
    test = "test.txt"
    test_dict = dict()
    with open(path) as f:
            for l in f.readlines():
                try:
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        test_dict[uid] = items
                except:
                    print(f"A user do not have items in the test data {l}") 
    
    
    test_df = (
        pd.Series(test_dict, name="ItemID")
        .explode()
        .reset_index()
        .rename(columns={"index": "UserID"})
    )
    # optional: ensure integers (in case of missing/strings)
    test_df["UserID"] = test_df["UserID"].astype(int)
    test_df["ItemID"] = test_df["ItemID"].astype(int)
    test_df["Rating "] = 1
    return test_df


print("\n" + "★"*44)
print("PREPARE DATA FOR NGCF MODEL".center(44))
print("★"*44 + "\n")

###### GOWALLA NGCF
path = "ngcf_data/gowalla"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/ngcf/gowalla/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)

train_users = len(train_df.UserID.unique())
test_users = len(test_df.UserID.unique())
print_dataset_info( train_users , test_users,   dataset_name = "GOWALLA")

######## AMAZON-BOOK NGCF
path = "ngcf_data/amazon-book"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/ngcf/amazon-book/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)
train_users = len(train_df.UserID.unique())
test_users = len(test_df.UserID.unique())
print_dataset_info( train_users , test_users,   dataset_name = "AMAZON-BOOK")


print("\n" + "★"*44)
print("PREPARE DATA FOR LIGHTGCN MODEL".center(44))
print("★"*44 + "\n")


######### LIGHTGCN GOWALLA
path = "lightgcn_data/gowalla"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/lightgcn/gowalla/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)

train_users = len(train_df.UserID.unique())
test_users = len(test_df.UserID.unique())
print_dataset_info( train_users , test_users,   dataset_name = "GOWALLA")


########## LIGHTGCN AMAZON-BOOK
path = "lightgcn_data/amazon-book"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/lightgcn/amazon-book/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)

train_users = len(train_df.UserID.unique())
test_users = len(test_df.UserID.unique())
print_dataset_info( train_users , test_users,   dataset_name = "AMAZON-BOOK")


############ LIGHTGCN YELP2018 
path = "lightgcn_data/yelp2018"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("elliot/data/lightgcn/yelp2018/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)

train_users = len(train_df.UserID.unique())
test_users = len(test_df.UserID.unique())
print_dataset_info( train_users , test_users,   dataset_name = "YELP2018")



print(r"""
  _____   ____   _   _  ______
 |  __ \ / __ \ | \ | ||  ____|
 | |  | | |  | ||  \| || |__
 | |  | | |  | || . ` ||  __|
 | |__| | |__| || |\  || |____
 |_____/ \____/ |_| \_||______|
""")
