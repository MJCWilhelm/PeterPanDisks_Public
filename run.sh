#!/bin/bash

EPE=("1e-8" "1e-9" "1e-10" "-1")
ALPHA=("1e-4" "1e-3")

mkdir data
cd src

for E in ${EPE[*]}; do
    for A in ${ALPHA[*]}; do
        #python run_disks --alpha $A --mdot $E
        mpiexec /data1/wilhelm/py_envs/amuse_dev/amuse/amuse.sh run_disks.py --alpha $A --mdot $E
    done
done
