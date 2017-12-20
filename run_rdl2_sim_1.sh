#!/bin/bash

python amber_gbsa.py \
       --steps 1000000 \
       --positions-fl simulations/rdl2_implicit_sim_01.dcd \
       --energies-fl simulations/rdl2_implicit_sim_01.energies.tsv \
       --output-state simulations/rdl2_impilict_sim_01.state \
       --steps-output 10000 \
       --temperature 300.0 \
       --damping 50.0 \
       --timestep 0.001 \
       --forcefield amber99sbildn \
       --pdb-file models/model1_fixed_moved2_eq1.pdb
