#!/bin/bash

for year in {2012..2018}; do
    python MCSonsetDetection.py \
        -p data/PIRATA/ \
        -t data/TOOCAN/AFRICA/$year \
        -o data/coldpools/${year}coldpools.csv
done