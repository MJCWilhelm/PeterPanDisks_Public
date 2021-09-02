#!/bin/bash

EPE=("1e-8" "1e-9" "1e-10" "-1")
ALPHA=("1e-4" "1e-3")

mkdir data
mkdir figures
cd src

for E in ${EPE[*]}; do
    for A in ${ALPHA[*]}; do
        python run_disks.py --alpha $A --mdot $E
    done
done

python plots_fried.py
python analyze_disks.py
