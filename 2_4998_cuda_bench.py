import os
import time
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier

# Execute from command line with
# syrupy.py -i 1 -t 2_4998_cuda --separator=, --no-align python3 2_4998_cuda_bench.py

# ------------------------------------------------------------------------------

num_train = 2
num_pred = 4998

nm = str(num_train) + "_" + str(num_pred) + "_cuda_summary.csv"

# ------------------------------------------------------------------------------

tr_dat = pd.read_csv("forested_train.csv")
te_dat = pd.read_csv("forested_test.csv")
all_dat = pd.concat([tr_dat, te_dat], ignore_index=True)

tr_ind = np.arange(0, num_train, 1)
te_ind = np.arange(0, num_pred, 1) + num_train + 1

tr_x = all_dat.iloc[tr_ind, 1:17]
tr_y = all_dat.iloc[tr_ind, 0]
te_x = all_dat.iloc[te_ind, 1:17]

# ------------------------------------------------------------------------------

tab_pfn = TabPFNClassifier(device="cuda")
tab_pfn.fit(tr_x, tr_y)

# ------------------------------------------------------------------------------

pred_start_time = time.perf_counter()

te_pred = tab_pfn.predict_proba(te_x)

pred_stop_time = time.perf_counter()

pred_time = pred_stop_time - pred_start_time

# ------------------------------------------------------------------------------

session_data = {
    "num_train": len(tr_x),
    "num_pred": len(te_x),
    "pred_time": pred_time,
    "pid": os.getpid(),
    "gpu": "cuda",
}

session_data = pd.DataFrame(session_data, index=[0])
session_data.to_csv(nm, index=False)
