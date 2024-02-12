#!/usr/bin/env bash

ex1 () {
    python3 image_example.py \
        -q ./QK001-QC.JPG \
        -qp ./QK001-QC.json \
        -qd 12 \
        -k ./QK001-KG.JPG \
        -kp ./QK001-KG.json \
        -kd 20 \
        --delta 0.005 \
        --epsilon 0.2 \
        --lower-bound 3 \
        --output "./QK001-viz.gif"
}

ex2 () {
    python3 image_example.py \
        -q ./QK010-QC.JPG \
        -qp ./QK010-QC.json \
        -qd 12 \
        -k ./QK010-KF.JPG \
        -kp ./QK010-KF.json \
        -kd 20 \
        --delta 0.005 \
        --epsilon 0.2 \
        --lower-bound 3 \
        --output "./QK010-viz.gif"
}

ex3 () {
    python3 image_example.py \
        -k ./050L_flipped.tif \
        -kp ./050L_flipped.json \
        -kd 2 \
        -q ./050LBFT11-21-22AGGV12-05-22MM.tif \
        -qp ./050LBFT11.json \
        -qd 2 \
        --delta 0.003 \
        --epsilon 0.5 \
        --lower-bound 10 \
        --check-ties \
        --output "./SC050L-viz.mp4"
}

ex3
