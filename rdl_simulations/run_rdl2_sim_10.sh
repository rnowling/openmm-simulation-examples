#!/bin/bash

python amber_gbsa.py \
       --steps 500000000 \
       --positions-fl simulations/rdl2_implicit_sim_10.dcd \
       --energies-fl simulations/rdl2_implicit_sim_10.energies.tsv \
       --output-state simulations/rdl2_implicit_sim_10.state \
       --input-state simulations/rdl2_implicit_sim_09.state \
       --steps-output 5000 \
       --temperature 300.0 \
       --damping 50.0 \
       --timestep 0.002 \
       --forcefield amber99sbildn \
       --pdb-file models/model1_fixed_moved2_eq1.pdb
