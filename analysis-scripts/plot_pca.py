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
from sklearn.decomposition import TruncatedSVD


def explained_variance_analysis(args):
    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "aligning frames"
    traj.superpose(traj)

    reshaped = traj.xyz.reshape(traj.n_frames,
                                traj.n_atoms * 3)

    print "Fitting SVD"
    svd = TruncatedSVD(n_components = args.n_components)
    svd.fit(reshaped)

    plt.clf()
    plt.grid(True)
    plt.plot(svd.explained_variance_ratio_, "m.-")
    plt.xlabel("Principal Component", fontsize=16)
    plt.ylabel("Explained Variance Ratio", fontsize=16)
    plt.ylim([0., 1.])
    fig_flname = os.path.join(args.figures_dir, "pca_explained_variance_ratios.png")
    plt.savefig(fig_flname,
                DPI=300)

def pairwise(iterable):
    iterable = iter(iterable)
    try:
        while True:
            a = next(iterable)
            b = next(iterable)
            yield a, b
    except StopIteration:
        pass
    
def plot_projections(args):
    if len(args.pairs) % 2 != 0:
        print "Error: PCs must be provided in pairs of 2"
        sys.exit(1)

    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "aligning frames"
    traj.superpose(traj)

    reshaped = traj.xyz.reshape(traj.n_frames,
                                traj.n_atoms * 3)

    print "Fitting SVD"
    svd = TruncatedSVD(n_components = args.n_components)
    projected = svd.fit_transform(reshaped)

    for p1, p2 in pairwise(args.pairs):
        fig_flname = os.path.join(args.figures_dir,
                                  "pca_projection_%s_%s.png" % (str(p1), str(p2)))
        plt.clf()
        plt.grid(True)
        plt.scatter(projected[:, p1],
                    projected[:, p2],
                    color="c",
                    marker="o",
                    edgecolor="k",
                    alpha=0.7)
        plt.xlabel("Principal Component %s" % p1, fontsize=16)
        plt.ylabel("Principal Component %s" % p2, fontsize=16)
        plt.savefig(fig_flname,
                    DPI=300)

    
def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--figures-dir",
                        type=str,
                        required=True,
                        help="Figure output directory")

    parser.add_argument("--pdb-file",
                        type=str,
                        required=True,
                        help="Input PDB file")

    parser.add_argument("--input-traj",
                        type=str,
                        required=True,
                        help="Input trajectory file")

    parser.add_argument("--n-components",
                        type=int,
                        required=True,
                        help="Number of PCs to compute")

    subparsers = parser.add_subparsers(dest="mode")
    eva_parser = subparsers.add_parser("explained-variance-analysis",
                                       help="Compute explained variances of PCs")

    plot_parser = subparsers.add_parser("plot-projections",
                                        help="Plot structures on projections")

    plot_parser.add_argument("--pairs",
                             type=int,
                             nargs="+",
                             required=True,
                             help="Pairs of PCs to plot")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if args.mode == "explained-variance-analysis":
        explained_variance_analysis(args)
    elif args.mode == "plot-projections":
        plot_projections(args)
    else:
        print "Unknown mode '%s'" % args.mode
        sys.exit(1)
