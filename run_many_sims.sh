#!/bin/bash

python_exec="/usr/bin/python3"
nside=512
#Create 500 simulations
for seed in {1000..1500}
do
    addqueue -q cmb -m 32 ${python_exec} pysm_componentspip2_temp.py ${seed}
    echo "/mnt/extraspace/susanna/SO/PySM-test-outputs/sim_constBeta_seed${seed}" >> list_sims.txt
done
