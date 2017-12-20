"""
Copyright 2017 Ronald J. Nowling

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np


def plot_end_to_end(args):
    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "aligning frames"
    backbone = traj.topology.select_atom_indices("minimal")
    traj.superpose(traj, atom_indices=backbone)

    if args.select_atoms:
        selection = traj.topology.select(args.select_atoms)
    else:
        selection = traj.topology.select("protein")

    print "computing distances"
    distances = []
    for frame in xrange(traj.n_frames):
        min_x = traj.xyz[frame, selection, 0].min()
        max_x = traj.xyz[frame, selection, 0].max()
        min_y = traj.xyz[frame, selection, 1].min()
        max_y = traj.xyz[frame, selection, 1].max()
        min_z = traj.xyz[frame, selection, 2].min()
        max_z = traj.xyz[frame, selection, 2].max()

        bottom = np.array([min_x, min_y, min_z])
        top = np.array([max_x, max_y, max_z])

        diff = bottom - top
        d = np.sqrt(np.dot(diff, diff))
        distances.append(d)

    plt.clf()
    plt.plot(np.arange(1, traj.n_frames + 1) * traj.timestep,
             distances)
    plt.xlabel("Time (ps)", fontsize=16)
    plt.ylabel("End-to-End Distance (nm)", fontsize=16)
    plt.savefig(args.figures_fl, PDI=300)

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--figures-fl",
                        type=str,
                        required=True,
                        help="Figure output filename")

    parser.add_argument("--pdb-file",
                        type=str,
                        required=True,
                        help="Input PDB file")

    parser.add_argument("--input-traj",
                        type=str,
                        required=True,
                        help="Input trajectory file")

    parser.add_argument("--select-atoms",
                        type=str,
                        help="Atom selection expression")

    return parser.parse_args()
    
if __name__ == "__main__":
    args = parseargs()

    plot_end_to_end(args)
