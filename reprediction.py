import random
import time
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier

# ------------------------------------------------------------------------------

tr_dat = pd.read_csv("forested_train.csv")
te_dat = pd.read_csv("forested_test.csv")
tr_x = tr_dat.iloc[:, 1:17]
tr_y = tr_dat.iloc[:, 0]

random.seed(4123)
tab_pfn = TabPFNClassifier(device="cuda")
tab_pfn.fit(tr_x, tr_y)
pred_start_time = time.perf_counter()
pred = tab_pfn.predict_proba(tr_x)
pred_stop_time = time.perf_counter()

pred_time = pred_stop_time - pred_start_time
print(pred_time)

pred = pd.DataFrame(pred)
pred.to_csv("training_set_predictions.csv", index=False)
