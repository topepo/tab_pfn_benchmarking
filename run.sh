#!/bin/bash
source tab_pfn/bin/activate  
syrupy.py -i 1 -t    2_4998_cuda --separator=, --no-align python3    2_4998_cuda_bench.py
syrupy.py -i 1 -t 2500_2500_cuda --separator=, --no-align python3 2500_2500_cuda_bench.py
syrupy.py -i 1 -t    4998_2_cuda --separator=, --no-align python3    4998_2_cuda_bench.py
python3 predictions_by_training_size_cuda.py
python3 predictions_by_num_cols_cuda.py
