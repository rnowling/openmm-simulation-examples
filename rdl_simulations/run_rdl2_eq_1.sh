#!/bin/bash

python amber_gbsa.py \
       --steps 1000000 \
       --positions-fl simulations/rdl2_implicit_eq1.dcd \
       --energies-fl simulations/rdl2_implicit_eq1.energies.tsv \
       --steps-output 10000 \
       --temperature 300.0 \
       --damping 10000.0 \
       --timestep 0.00001 \
       --forcefield amber99sbildn \
       --pdb-file models/model1_fixed_moved2.pdb \
       --minimize
