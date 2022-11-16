import pickle
import numpy as np
import pandas as pd


def load_raw_data():
    df = pd.read_csv("./data/ILINet.csv")
    df = df[["REGION", "YEAR", "WEEK", "% WEIGHTED ILI"]]
    df = df[(df["YEAR"] >= 2004) | ((df["YEAR"] == 2003) & (df["WEEK"] >= 20))]

    return df

def get_dataset(year: int, region: str, df=None):
    ans = df[
        ((df["YEAR"] == year) & (df["WEEK"] >= 20))
        | ((df["YEAR"] == year + 1) & (df["WEEK"] <= 20))
    ]
    return ans[ans["REGION"] == region]["% WEIGHTED ILI"]


def one_hot(idx, dim):
    ans = np.zeros(dim, dtype="float32")
    ans[idx] = 1.0
    return ans


def save_data(obj, filepath):
    with open(filepath, "wb") as fl:
        pickle.dump(obj, fl)

def extract_data(train_seasons, val_seasons, test_seasons, regions, city_idx):
    df = load_raw_data()
    
    full_x = np.array(
        [
            np.array(get_dataset(s, r, df), dtype="float32")[-53:]
            for s in train_seasons
            for r in regions
        ]  # [(s1,r1),(s1,r2),...(s2,r1),(s2,r2),...] (154, 53)
    )
    full_meta = np.array([one_hot(city_idx[r], len(city_idx)) for s in train_seasons for r in regions])
    full_y = full_x.argmax(-1)
    full_x = full_x[:, :, None]

    full_x_val = np.array(
        [
            np.array(get_dataset(s, r, df), dtype="float32")[-53:]
            for s in val_seasons
            for r in regions
        ]  
    )
    full_meta_val = np.array([one_hot(city_idx[r], len(city_idx)) for s in val_seasons for r in regions])
    full_y_val = full_x_val.argmax(-1)
    full_x_val = full_x_val[:, :, None]

    full_x_test = np.array(
        [
            np.array(get_dataset(s, r, df), dtype="float32")[:40] # TODO: remove data when covid happened
            for s in test_seasons
            for r in regions
        ]
    )
    full_meta_test = np.array([one_hot(city_idx[r], len(city_idx)) for s in test_seasons for r in regions])
    full_y_test = full_x_test.argmax(-1)
    full_x_test = full_x_test[:, :, None]

    train_info = full_x, full_y, full_meta
    val_info = full_x_val, full_y_val, full_meta_val
    test_info = full_x_test, full_y_test, full_meta_test
    return train_info, val_info, test_info

def create_dataset(full_meta, full_x, week_ahead=None):
    metas, seqs, y = [], [], []
    for meta, seq in zip(full_meta, full_x): # (154, 11), (154, 53)
        for i in range(20, full_x.shape[1]):
            metas.append(meta)
            seqs.append(seq[: i - week_ahead + 1])
            y.append(seq[i])
    return np.array(metas, dtype="float32"), seqs, np.array(y, dtype="float32")


