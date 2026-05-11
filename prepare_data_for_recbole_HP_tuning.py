import pandas as pd
from pathlib import Path
from helper_functions import *


def create_df_in_recbole_format(_dict):
        df = (
        pd.Series(_dict, name="item_id:token")
        .explode()
        .reset_index()
        .rename(columns={"index": "user_id:token"})
        )

        # optional: ensure integers (in case of missing/strings)
        df["user_id:token"] = df["user_id:token"].astype(int)
        df["item_id:token"] = df["item_id:token"].astype(int)
        df["rating:float"] = 1

        return df

def training_data(path):
    train_dict = dict()
    valid_dict = dict()
    try:
        with open(path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) > 3:

                        items_train = [int(i) for i in l[1:-1]]
                        item_valid = [int(l[-1])]
                        uid = int(l[0])
                        train_dict[uid] = items_train
                        valid_dict[uid] = item_valid
                    else:
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        train_dict[uid] = items

    except:
        print(f"A user do not have items in the training data {l}")
    
    train_df = create_df_in_recbole_format(train_dict)
    valid_df = create_df_in_recbole_format(valid_dict)
    
    return train_df, valid_df

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
print_dataset_name(dataset_name="GOWALLA")

train_name = "gowalla.train.inter"
valid_name = "gowalla.valid.inter"
test_name = "gowalla.test.inter"

path = "ngcf_data/gowalla"
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")


path = Path("RecBole/data/ngcf/tuning/gowalla/")
path.mkdir(parents=True, exist_ok=True)


train_df.to_csv(path / train_name, sep = "\t", index = False)
valid_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
valid_users = len(valid_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())
print_dataset_info( train_users , test_users, valid_users,  dataset_name = "GOWALLA")


######## AMAZON-BOOK NGCF
print_dataset_name(dataset_name="AMAZON-BOOK")

train_name = "amazon-book.train.inter"
valid_name = "amazon-book.valid.inter"
test_name = "amazon-book.test.inter"


path = "ngcf_data/amazon-book"
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("RecBole/data/ngcf/tuning/amazon-book/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / train_name, sep = "\t", index = False)
valid_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
valid_users = len(valid_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())

print_dataset_info( train_users , test_users, valid_users,  dataset_name = "AMAZON-BOOK")




print("\n" + "★"*44)
print("PREPARE DATA FOR LIGHTGCN MODEL".center(44))
print("★"*44 + "\n")


######### LIGHTGCN GOWALLA
print_dataset_name(dataset_name="GOWALLA")
train_name = "gowalla.train.inter"
valid_name = "gowalla.valid.inter"
test_name = "gowalla.test.inter"

path = "lightgcn_data/gowalla"
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("RecBole/data/lightgcn/tuning/gowalla/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / train_name, sep = "\t", index = False)
valid_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
valid_users = len(valid_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())
print_dataset_info( train_users , test_users, valid_users,   dataset_name = "GOWALLA")

########## LIGHTGCN AMAZON-BOOK
print_dataset_name(dataset_name="AMAZON-BOOK")
train_name = "amazon-book.train.inter"
valid_name = "amazon-book.valid.inter"
test_name = "amazon-book.test.inter"

path = "lightgcn_data/amazon-book"
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("RecBole/data/lightgcn/tuning/amazon-book/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / train_name, sep = "\t", index = False)
valid_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
valid_users = len(valid_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())
print_dataset_info( train_users , test_users, valid_users,   dataset_name = "AMAZON-BOOK")

############ LIGHTGCN YELP2018 
print_dataset_name(dataset_name="YELP2018")
train_name = "yelp2018.train.inter"
valid_name = "yelp2018.valid.inter"
test_name = "yelp2018.test.inter"

path = "lightgcn_data/yelp2018"
path = Path(path)
train_df, valid_df = training_data(path / "train.txt")
test_df = testing_data(path / "test.txt")

path = Path("RecBole/data/lightgcn/tuning/yelp2018/")
path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(path / train_name, sep = "\t", index = False)
valid_df.to_csv(path / valid_name, sep = "\t", index = False)
test_df.to_csv(path / test_name, sep = "\t", index = False)

train_users = len(train_df["user_id:token"].unique())
valid_users = len(test_df["user_id:token"].unique())
test_users = len(test_df["user_id:token"].unique())
print_dataset_info( train_users , test_users, valid_users,  dataset_name = "AMAZON-BOOK")

print(r"""
  _____   ____   _   _  ______
 |  __ \ / __ \ | \ | ||  ____|
 | |  | | |  | ||  \| || |__
 | |  | | |  | || . ` ||  __|
 | |__| | |__| || |\  || |____
 |_____/ \____/ |_| \_||______|
""")
