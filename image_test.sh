#!/usr/bin/env bash

python3 image_example.py \
    -q ./QK001-QC.JPG \
    -qp ./QK001-QC.json \
    -qd 12 \
    -k ./QK001-KG.JPG \
    -kp ./QK001-KG.json \
    -kd 18 \
    --delta 0.01 \
    --epsilon 0.03 \
    --lower-bound 10 \
    --output "./QK001-viz.mp4"
