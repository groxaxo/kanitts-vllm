#!/bin/bash
# Startup script for KaniTTS Parallel Deployment

# Kill existing servers
pkill -f "python server"

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kanitts-vllm

# Start Spanish Server
echo "Starting Spanish Server on 8001..."
nohup python server_spanish.py > spanish_server.log 2>&1 &
PID_ES=$!
echo "Spanish Server PID: $PID_ES"

# Wait for Spanish Server to initialize
echo "Waiting for Spanish Server to be ready..."
timeout 60s bash -c 'until curl -s http://localhost:8001/health > /dev/null; do sleep 2; done'

# Start English Server
echo "Starting English Server on 8002..."
nohup python server_english.py > english_server.log 2>&1 &
PID_EN=$!
echo "English Server PID: $PID_EN"

# Wait for English Server to initialize
echo "Waiting for English Server to be ready..."
timeout 60s bash -c 'until curl -s http://localhost:8002/health > /dev/null; do sleep 2; done'

# Start Router
echo "Starting Router on 8000..."
nohup python server_router.py > router_server.log 2>&1 &
PID_ROUTER=$!
echo "Router PID: $PID_ROUTER"

echo "Deployment Complete!"
echo "Router: http://localhost:8000"
echo "Spanish: http://localhost:8001"
echo "English: http://localhost:8002"
