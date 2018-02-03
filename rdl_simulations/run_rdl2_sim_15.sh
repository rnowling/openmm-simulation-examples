#!/bin/bash

python ../amber_gbsa.py \
       --steps 1000000000 \
       --positions-fl ~/Data/rdl/simulations/rdl2_implicit_sim_15.dcd \
       --energies-fl ~/Data/rdl/simulations/rdl2_implicit_sim_15.energies.tsv \
       --output-state rdl2_implicit_sim_15.state \
       --input-state rdl2_implicit_sim_14.state \
       --steps-output 5000 \
       --temperature 300.0 \
       --damping 50.0 \
       --timestep 0.002 \
       --forcefield amber99sbildn \
       --pdb-file model1_fixed_moved2_eq1.pdb
