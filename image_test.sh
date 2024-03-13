#!/usr/bin/env bash

ex1 () {
    python3 image_example.py \
        -q ./examples/QK001-QC.JPG \
        -qp ./examples/QK001-QC.json \
        -qd 12 \
        -k ./examples/QK001-KG.JPG \
        -kp ./examples/QK001-KG.json \
        -kd 20 \
        --delta 0.005 \
        --epsilon 0.2 \
        --lower-bound 3 \
        --output "./examples/QK001-viz.gif"
}

ex2 () {
    python3 image_example.py \
        -q ./examples/QK010-QC.JPG \
        -qp ./examples/QK010-QC.json \
        -qd 12 \
        -k ./examples/QK010-KF.JPG \
        -kp ./examples/QK010-KF.json \
        -kd 20 \
        --delta 0.005 \
        --epsilon 0.2 \
        --lower-bound 3 \
        --output "./examples/QK010-viz.gif"
}

ex3 () {
    python3 image_example.py \
        -k ./examples/050L_flipped.tif \
        -kp ./examples/050L_flipped.json \
        -kd 5 \
        -q ./examples/050LBFT11-21-22AGGV12-05-22MM.tif \
        -qp ./examples/050LBFT11.json \
        -qd 5 \
        -min 0.8 \
        -max 1.5 \
        --delta 0.008 \
        --epsilon 0.1 \
        --lower-bound 10 \
        --output "./examples/SC050L-viz.mp4"
}

ex4 () {
    python3 image_example.py \
        -k ./examples/698RH1_flipped.tif \
        -kp ./examples/698RH1_flipped.json \
        -kd 12 \
        -q ./examples/698RB3_E.tif \
        -qp ./examples/698RB3_E.json \
        -qd 15 \
        -min 0.75 \
        -max 1.0 \
        --delta 0.008 \
        --epsilon 0.05 \
        --lower-bound 10 \
        --output "./examples/WVU-698RB3-viz.mp4"
}

ex2
