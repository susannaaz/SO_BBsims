#!/bin/bash

python_exec="/usr/bin/python3"
nside=512
for seed in {1100..1500}
do
    addqueue -q cmb -m 32 ${python_exec} pysm_componentspip2_temp.py ${seed}
    echo "/mnt/extraspace/susanna/SO/PySM-test-outputs/sim_constBeta_seed${seed}" >> list_sims4.txt
done
