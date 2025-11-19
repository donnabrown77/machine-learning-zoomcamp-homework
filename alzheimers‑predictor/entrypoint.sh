#!/bin/sh
python predict.py &
echo "Waiting for API to be ready..."
while ! curl -s http://localhost:9696/health > /dev/null; do
    sleep 1
done
echo "API is ready!"
python serve.py
