#!/bin/bash

# Activate conda environment
source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate kanitts

# Start the server
python server.py
