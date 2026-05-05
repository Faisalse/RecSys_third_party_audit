import pandas as pd
from pathlib import Path
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
        pd.Series(train_dict, name="item_id:token")
        .explode()
        .reset_index()
        .rename(columns={"index": "user_id:token"})
    )
    # optional: ensure integers (in case of missing/strings)
    train_df["user_id:token"] = train_df["user_id:token"].astype(int)
    train_df["item_id:token"] = train_df["item_id:token"].astype(int)
    train_df["rating:float"] = 1
    return train_df

def testing_data(path):
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
        pd.Series(test_dict, name="item_id:token")
        .explode()
        .reset_index()
        .rename(columns={"index": "user_id:token"})
    )
    # optional: ensure integers (in case of missing/strings)
    test_df["user_id:token"] = test_df["user_id:token"].astype(int)
    test_df["item_id:token"] = test_df["item_id:token"].astype(int)
    test_df["rating:float"] = 1
    return test_df



print("\n" + "★"*44)
print("RECBOLE: PREPARE DATA FOR NGCF MODEL".center(44))
print("★"*44 + "\n")

###### GOWALLA NGCF
train_name = "gowalla.train.inter"
valid_name = "gowalla.valid.inter"
test_name = "gowalla.test.inter"

path = "ngcf_data/gowalla"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("RecBole/dataset/ngcf/gowalla/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / train_name, sep = "\t", index = False)
test_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())
print_dataset_info( train_users , test_users,   dataset_name = "GOWALLA")


######## AMAZON-BOOK NGCF
train_name = "amazon-book.train.inter"
valid_name = "amazon-book.valid.inter"
test_name = "amazon-book.test.inter"


path = "ngcf_data/amazon-book"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("RecBole/dataset/ngcf/amazon-book/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / train_name, sep = "\t", index = False)
test_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())
print_dataset_info( train_users , test_users,   dataset_name = "AMAZON-BOOK")


print("\n" + "★"*44)
print("PREPARE DATA FOR LIGHTGCN MODEL".center(44))
print("★"*44 + "\n")


######### LIGHTGCN GOWALLA
train_name = "gowalla.train.inter"
valid_name = "gowalla.valid.inter"
test_name = "gowalla.test.inter"

path = "lightgcn_data/gowalla"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("RecBole/dataset/lightgcn/gowalla/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / train_name, sep = "\t", index = False)
test_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())
print_dataset_info( train_users , test_users,   dataset_name = "GOWALLA")

########## LIGHTGCN AMAZON-BOOK
train_name = "amazon-book.train.inter"
valid_name = "amazon-book.valid.inter"
test_name = "amazon-book.test.inter"

path = "lightgcn_data/amazon-book"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("RecBole/dataset/lightgcn/amazon-book/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / train_name, sep = "\t", index = False)
test_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())
print_dataset_info( train_users , test_users,   dataset_name = "AMAZON-BOOK")

############ LIGHTGCN YELP2018 
train_name = "yelp2018.train.inter"
valid_name = "yelp2018.valid.inter"
test_name = "yelp2018.test.inter"

path = "lightgcn_data/yelp2018"
path = Path(path)
train_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("RecBole/dataset/lightgcn/yelp2018/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / train_name, sep = "\t", index = False)
test_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())
print_dataset_info( train_users , test_users,   dataset_name = "AMAZON-BOOK")

print(r"""
  _____   ____   _   _  ______
 |  __ \ / __ \ | \ | ||  ____|
 | |  | | |  | ||  \| || |__
 | |  | | |  | || . ` ||  __|
 | |__| | |__| || |\  || |____
 |_____/ \____/ |_| \_||______|
""")
