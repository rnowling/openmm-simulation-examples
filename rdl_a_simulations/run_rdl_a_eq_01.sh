#!/bin/bash

mkdir -p ~/scratch/rdl/rdl_a_simulations
#       --input-state simulations/rdl2_implicit_sim_09.state \

python ../amber_gbsa.py \
       --steps 500000000 \
       --positions-fl ~/scratch/rdl/rdl_a_simulations/rdl_a_implicit_eq_01.dcd \
       --energies-fl ~/scratch/rdl/rdl_a_simulations/rdl_a_implicit_eq_01.energies.tsv \
       --minimize \
       --steps-output 5000 \
       --temperature 300.0 \
       --damping 10000.0 \
       --timestep 0.00001 \
       --forcefield amber99sbildn \
       --pdb-file ../models/rdl_a_homology_model_fixed.pdb
