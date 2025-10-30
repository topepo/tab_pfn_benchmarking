import random
import time
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier

# ------------------------------------------------------------------------------

tr_dat = pd.read_csv("forested_train.csv")
te_dat = pd.read_csv("forested_test.csv")
te_x = te_dat.iloc[:, 1:17]
te_y = te_dat.iloc[:, 0]

# ------------------------------------------------------------------------------

total_train = len(tr_dat)
num_train = 100

low_seq = np.arange(0, 50, 1)
log_seq = np.array(
    [
        63,
        80,
        102,
        130,
        166,
        212,
        268,
        342,
        434,
        554,
        704,
        896,
        1140,
        1450,
        1846,
        2348,
        2986,
        3798,
        4830,
    ]
)
size_seq = np.concatenate((low_seq, log_seq))

# ------------------------------------------------------------------------------

# Initially pick 1 of each class so that the model can run

base_x = tr_dat.iloc[0:2, 1:17]
base_y = tr_dat.iloc[0:2, 0]

random.seed(4123)
for size in size_seq:
    tr_ind = np.random.randint(0, total_train, size=size)
    tr_x = tr_dat.iloc[tr_ind, 1:17]
    tr_y = tr_dat.iloc[tr_ind, 0]

    tr_x = pd.concat([base_x, tr_x], axis=0)
    tr_y = pd.concat([base_y, tr_y], axis=0)

    tab_pfn = TabPFNClassifier()  # dev = cuda
    tab_pfn.fit(tr_x, tr_y)

    ###

    pred_start_time = time.perf_counter()
    pred = tab_pfn.predict_proba(te_x)
    pred_stop_time = time.perf_counter()

    pred_time = pred_stop_time - pred_start_time
    pred_time = {"num_train": len(tr_x), "time": pred_time}
    pred_time = pd.DataFrame(pred_time, index=[0])

    ###

    pred = pd.DataFrame(pred, columns=[".pred_No", ".pred_Yes"])
    pred = pd.concat([pred, te_y], axis=1)
    pred["num_train"] = len(tr_x)

    if size == 0:
        all_pred = pred
        all_times = pred_time
    else:
        all_pred = pd.concat([all_pred, pred], axis=0)
        all_times = pd.concat([all_times, pred_time], axis=0)

all_pred.to_csv("predictions_by_training_size.csv", index=False)
all_times.to_csv("predictions_by_training_size_times.csv", index=False)
