#!/bin/bash
conda activate ros_humble
python follow_person.py \
    --ip 10.0.11.162 \
    --port 2000 \
    --rtsp-url rtsp://admin:admin@10.0.11.162:554/live/av0
