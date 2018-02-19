#!/bin/bash

mkdir -p ~/scratch/rdl/rdl_a_simulations

python amber_gbsa.py \
       --steps 1000000000 \
       --positions-fl ~/scratch/rdl/rdl_a_simulations/rdl_a_implicit_sim_01.dcd \
       --energies-fl ~/scratch/rdl/rdl_a_simulations/rdl_a_implicit_sim_01.energies.tsv \
       --output-state rdl_a_implicit_sim_01.state \
       #       --input-state simulations/rdl2_implicit_sim_09.state \
       --minimize \
       --steps-output 5000 \
       --temperature 300.0 \
       --damping 50.0 \
       --timestep 0.002 \
       --forcefield amber99sbildn \
       --pdb-file models/rdl_a_homology_model.pdb
