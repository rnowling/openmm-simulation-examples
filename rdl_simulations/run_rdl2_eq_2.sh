#!/bin/bash

python amber_gbsa.py \
       --steps 1000000 \
       --positions-fl simulations/rdl2_implicit_eq2.dcd \
       --energies-fl simulations/rdl2_implicit_eq2.energies.tsv \
       --steps-output 10000 \
       --temperature 300.0 \
       --damping 1000.0 \
       --timestep 0.0001 \
       --forcefield amber99sbildn \
       --pdb-file models/model1_fixed_moved2_eq1.pdb \
       --minimize
