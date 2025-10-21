import random
import time
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier

# ------------------------------------------------------------------------------

tr_dat = pd.read_csv("forested_train.csv")
te_dat = pd.read_csv("forested_test.csv")

tr_x = tr_dat.iloc[0:100, 1:17]
tr_y = tr_dat.iloc[0:100, 0]

te_x = te_dat.iloc[:, 1:17]
te_y = te_dat.iloc[:, 0]

num_te = len(te_x)

# ------------------------------------------------------------------------------

noise_seq = np.arange(0, 70, 1) * 5

# ------------------------------------------------------------------------------

random.seed(4123)
for noise_cols in noise_seq:
    if noise_cols > 0:
        noise_names = [f"Column_{i + 1}" for i in range(noise_cols)]

        tr_rnd = np.random.rand(100, noise_cols)
        tr_rnd = pd.DataFrame(tr_rnd, columns=noise_names)
        tr_rnd = tr_rnd.reset_index(drop=True)

        te_rnd = np.random.rand(num_te, noise_cols)
        te_rnd = pd.DataFrame(te_rnd, columns=noise_names)
        te_rnd = te_rnd.reset_index(drop=True)

        tr_noise = pd.concat([tr_rnd, tr_x], axis=1)
        te_noise = pd.concat([te_rnd, te_x], axis=1)
    else:
        tr_noise = tr_x
        te_noise = te_x

    tab_pfn = TabPFNClassifier()
    tab_pfn.fit(tr_noise, tr_y)

    ###

    pred_start_time = time.perf_counter()
    pred = tab_pfn.predict_proba(te_noise)
    pred_stop_time = time.perf_counter()

    pred_time = pred_stop_time - pred_start_time
    pred_time = {"num_cols": te_noise.shape[1], "time": pred_time, "gpu": "none"}
    pred_time = pd.DataFrame(pred_time, index=[0])

    if noise_cols == 0:
        all_times = pred_time
    else:
        all_times = pd.concat([all_times, pred_time], axis=0)

all_times.to_csv("predictions_by_num_cols_times.csv", index=False)
