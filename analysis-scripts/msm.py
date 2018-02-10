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
from itertools import combinations
import os
import sys

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import mdtraj as md
import networkx as nx
import numpy.linalg as LA
import numpy as np
from scipy import stats
from sklearn.cluster import k_means
from sklearn.externals import joblib


MODEL_TYPE_KEY = "model-type"
PCA_MODEL = "pca"
SVD_MODEL = "svd"
ICA_MODEL = "ica"
TICA_MODEL = "tica"
MODEL_KEY = "model"
PROJECTION_KEY = "projected-coordinates"

class MarkovModel(object):
    def __init__(self, n_states, timestep, stride):
        self.n_states = n_states
        self.timestep = timestep
        self.stride = stride

    def fit(self, frames):
        _, self.labels, inertia = k_means(frames,
                                          self.n_states,
                                          n_jobs=-2,
                                          tol=0.00001,
                                          n_init=25)

        # we want to re-order the states for readability so we:
        # 1. build the count matrix
        # 2. perform a DFS from the first state
        # 3. map the old states to the new states
        # 4. re-label the frames
        # 5. re-compute the count matrix
        #
        # I know this could be done by re-ordering the rows / cols
        # of the count matrix but I'm too lazy to figure that out
        # right now.

        counts = np.zeros((self.n_states,
                           self.n_states))
        for i, from_ in enumerate(self.labels):
            j = i + self.stride
            if j < len(self.labels):
                to_ = self.labels[j]
                counts[to_, from_] += 1

        try:
            G = nx.DiGraph(counts.T)
            forward_mapping = list(nx.dfs_preorder_nodes(G, source=self.labels[0]))
            print forward_mapping
            if len(forward_mapping) != self.n_states:
                raise Exception("Unable to re-order states for readability")

            reverse_mapping = [-1] * self.n_states
            for new_idx, old_idx in enumerate(forward_mapping):
                reverse_mapping[old_idx] = new_idx

            new_labels = self.labels.copy()
            for idx, orig_label in enumerate(self.labels):
                new_label = reverse_mapping[orig_label]
                new_labels[idx] = new_label
            self.labels = new_labels

            counts = np.zeros((self.n_states,
                           self.n_states))
            for i, from_ in enumerate(self.labels):
                j = i + self.stride
                if j < len(self.labels):
                    to_ = self.labels[j]
                    counts[to_, from_] += 1
        except Exception as ex:
            print ex

        self.obs_pop_counts = np.zeros(self.n_states,
                                  dtype=np.int)
        for idx in self.labels:
            self.obs_pop_counts[idx] += 1

        # for prettier printing
        print counts.astype(np.int32)

        # force symmetry
        self.sym_counts = 0.5 * (counts + counts.T)

        # normalize columns
        self.transitions = self.sym_counts / self.sym_counts.sum(axis=1)[:, None]

        # get left eigenvectors
        u, v = LA.eig(self.transitions.T)

        # re-order in descending order
        sorted_idx = np.argsort(u)[::-1]    
        u = u[sorted_idx]
        v = v[:, sorted_idx]

        self.timescales = - self.timestep * self.stride / np.log(u[1:])
        self.equilibrium_dist = v[:, 0] / v[:, 0].sum()
        self.v = v[:, 1:]

        print self.equilibrium_dist
        print self.obs_pop_counts / np.float(self.obs_pop_counts.sum())

        eq_vec = v[:, 0].reshape(1, -1)
        print np.dot(eq_vec.T, eq_vec)

def sweep_clusters(args):
    data = joblib.load(args.model_file)
    projected = data[PROJECTION_KEY]

    print "Model type", data[MODEL_TYPE_KEY]

    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    inertia_values = []
    for k in args.n_clusters:
        print "Clustering with %s states" % k
        _, _, inertia = k_means(projected[:, args.dimensions],
                                k,
                                n_jobs=-2)
        inertia_values.append(inertia)

    plt.plot(args.n_clusters,
             inertia_values,
             "k.-")
    plt.xlabel("Number of Clusters", fontsize=16)
    plt.ylabel("Inertia", fontsize=16)

    fig_flname = os.path.join(args.figures_dir,
                              "cluster_inertia")
    for dim in args.dimensions:
        fig_flname += "_%s" % dim
    fig_flname += ".png"

    plt.savefig(fig_flname,
                DPI=300)

def sweep_lag_times(args):
    data = joblib.load(args.model_file)
    projected = data[PROJECTION_KEY]

    print "Model type", data[MODEL_TYPE_KEY]

    timescales = []
    data = projected[:, args.dimensions]
    for stride in args.strides:
        print "Training MSM with with %s states and stride %s" % (args.n_states,
                                                                  stride)
        msm = MarkovModel(args.n_states,
                          args.timestep,
                          stride)
        msm.fit(data)
        timescales.append(msm.timescales)

    timescales = np.array(timescales)

    lag_times = [args.timestep * stride for stride in args.strides]
    n_timescales = timescales.shape[1]
    for i in xrange(n_timescales):
        print i, timescales[:, i]
        plt.semilogy(lag_times,
                     timescales[:, i],
                     "k.-")
    
    plt.xlabel("Lag Time (ns)", fontsize=16)
    plt.ylabel("Timescale (ns)", fontsize=16)

    plt.savefig(args.figure_fl,
                DPI=300)

def train_model(args):
    data = joblib.load(args.model_file)
    projected = data[PROJECTION_KEY]

    print "Model type", data[MODEL_TYPE_KEY]

    data = projected[:, args.dimensions]
    msm = MarkovModel(args.n_states,
                      args.timestep,
                      args.stride)
    msm.fit(data)

    joblib.dump(msm, args.msm_model_file)

def plot_fluxes(args):
    msm = joblib.load(args.msm_model_file)
    G = nx.DiGraph(msm.transitions)
    n_fluxes = msm.v.shape[-1]
    default_size = 300.0
    node_size = []
    expected = 1.0 / msm.n_states
    for p in msm.equilibrium_dist:
        node_size.append(default_size * (p / expected))
    for i in xrange(n_fluxes):
        colors = []
        for j in xrange(msm.n_states):
            if msm.v[j, i] >= 0.0:
                colors.append(0.9)
            else:
                colors.append(0.1)
        plt.clf()
        nx.draw(G,
                pos=nx.nx_pydot.pydot_layout(G, prog="neato"),
                cmap=plt.get_cmap('Vega20c'),
                node_color=colors,
                node_size=node_size,
                vmin=0.0,
                vmax=1.0,
                with_labels=True)
        flname = os.path.join(args.figures_dir,
                              "flux_%s.png" % (i + 1))
        plt.savefig(flname,
                    DPI=300)

def plot_state_timeseries(args):
    msm = joblib.load(args.msm_model_file)

    times = np.arange(1, len(msm.labels) + 1) * msm.timestep
    plt.plot(times,
             msm.labels,
             "k")

    plt.xlabel("Time (ns)", fontsize=16)
    plt.ylabel("State", fontsize=16)
    plt.grid(True)
    plt.savefig(args.figure_fl,
                DPI=300)
        
def plot_msm_network(args):
    msm = joblib.load(args.msm_model_file)

    G = nx.DiGraph(msm.transitions)
    
    default_size = 300.0
    node_size = []
    if args.scale_size == "observed-populations":
        total = msm.obs_pop_counts.sum()
        for p in msm.obs_pop_counts:
            node_size.append(p * default_size / total)
    elif args.scale_size == "equilibrium-populations":
        expected = 1.0 / msm.n_states
        for p in msm.equilibrium_dist:
            node_size.append(default_size * (p / expected))
    else:
        node_size = None

    nx.draw(G,
            pos=nx.nx_pydot.pydot_layout(G, prog="neato"),
            node_size=node_size,
            arrowstyle="->",
            with_labels=True,
            vmin=0.0,
            vmax=1.0,
            node_color=[0.9] * nx.number_of_nodes(G),
            cmap=plt.get_cmap("Vega20c"))

    plt.savefig(args.figure_fl,
                DPI=300)

def test_residue_dihedral_distributions(phi_1, psi_1, phi_2, psi_2):
    n_bins = 10
    bins = np.linspace(-np.pi, np.pi, num=n_bins + 1)

    # last residue has no phi / psi angles    
    n_residues = phi_1.shape[1]
    residue_pvalues = []
    for resid in xrange(n_residues):
        dist_1, _, _ = np.histogram2d(phi_1[:, resid],
                                      psi_1[:, resid],
                                      bins = [bins, bins])

        dist_2, _, _ = np.histogram2d(phi_2[:, resid],
                                      psi_2[:, resid],
                                      bins = [bins, bins])

        # fudge factor to ensure that no bins are empty
        dist_1 += 1

        freq_1 = (dist_1 / np.sum(dist_1)).flatten()

        freq_2 = (dist_2 / np.sum(dist_2)).flatten()

        G = 0
        for i in xrange(freq_1.shape[0]):
            # skip over empty bins
            if freq_2[i] > 0.0:
                G += freq_2[i] * np.log(freq_2[i] / freq_1[i])
        G *= 2 * dist_2.size

        df = (n_bins - 1) * (n_bins - 1)
        p = stats.chi2.sf(G, df)

        residue_pvalues.append((resid + 1, p))

    residue_pvalues.sort(key = lambda t: t[-1])
    return residue_pvalues

def compare_dihedral_distributions(args):
    msm = joblib.load(args.msm_model_file)
    

    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "computing dihedrals"
    _, phi_angles = md.compute_phi(traj,
                                   periodic=False)
    _, psi_angles = md.compute_psi(traj,
                                   periodic=False)

    for state_1 in xrange(msm.n_states - 1):
        state_1_frames = [idx for idx, state in enumerate(msm.labels) \
                          if state == state_1]
        phi_1 = phi_angles[state_1_frames, :]
        psi_1 = psi_angles[state_1_frames, :]

        for state_2 in xrange(state_1 + 1, msm.n_states):
            state_2_frames = [idx for idx, state in enumerate(msm.labels) \
                              if state == state_2]
            
            if msm.sym_counts[state_1, state_2] > 0:
                print "Testing State %s vs State %s" % (state_1, state_2)
                
                phi_2 = phi_angles[state_2_frames, :]
                psi_2 = psi_angles[state_2_frames, :]

                residue_pvalues = test_residue_dihedral_distributions(phi_1,
                                                                      psi_1,
                                                                      phi_2,
                                                                      psi_2)

                flname = os.path.join(args.output_dir,
                                      "state_dihedral_tests_%s_%s.tsv" % (state_1,
                                                                          state_2))
                with open(flname, "w") as fl:
                    for resid, pvalue in residue_pvalues:
                        fl.write("%s\t%s\n" % (resid, pvalue))


def parseargs():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    cluster_sweep_parser = subparsers.add_parser("sweep-clusters",
                                                 help="Calculate inertia for different numbers of states")

    cluster_sweep_parser.add_argument("--figures-dir",
                                      type=str,
                                      required=True,
                                      help="Figure output directory")

    cluster_sweep_parser.add_argument("--dimensions",
                                      type=int,
                                      nargs="+",
                                      required=True,
                                      help="Dimensions to use in clustering")

    cluster_sweep_parser.add_argument("--n-clusters",
                                      type=int,
                                      nargs="+",
                                      required=True,
                                      help="Number of clusters to use")
    
    cluster_sweep_parser.add_argument("--model-file",
                                      type=str,
                                      required=True,
                                      help="File from which to load model")

    lag_time_sweep_parser = subparsers.add_parser("sweep-lag-times",
                                                  help="Sweep lag times")

    lag_time_sweep_parser.add_argument("--dimensions",
                                       type=int,
                                       nargs="+",
                                       required=True,
                                       help="Dimensions to use in clustering")

    lag_time_sweep_parser.add_argument("--strides",
                                       type=int,
                                       nargs="+",
                                       required=True,
                                       help="Strides to use when computing transitions")

    lag_time_sweep_parser.add_argument("--n-states",
                                       type=int,
                                       required=True,
                                       help="Number of states to use")
    
    lag_time_sweep_parser.add_argument("--model-file",
                                       type=str,
                                       required=True,
                                       help="File from which to load model")

    lag_time_sweep_parser.add_argument("--timestep",
                                       type=float,
                                       required=True,
                                       help="Elapsed time in ns between frames")
    
    lag_time_sweep_parser.add_argument("--figure-fl",
                                       type=str,
                                       help="Plot timescales",
                                       required=True)

    train_parser = subparsers.add_parser("train-model",
                                         help="Train and save a model")

    train_parser.add_argument("--dimensions",
                              type=int,
                              nargs="+",
                              required=True,
                              help="Dimensions to use in clustering")

    train_parser.add_argument("--stride",
                              type=int,
                              required=True,
                              help="Strides to use when computing transitions")

    train_parser.add_argument("--n-states",
                              type=int,
                              required=True,
                              help="Number of states to use")
    
    train_parser.add_argument("--model-file",
                              type=str,
                              required=True,
                              help="File from which to load model")

    train_parser.add_argument("--msm-model-file",
                              type=str,
                              required=True,
                              help="File to which to save MSM model")

    train_parser.add_argument("--timestep",
                              type=float,
                              required=True,
                              help="Elapsed time in ns between frames")

    draw_parser = subparsers.add_parser("draw-network",
                                        help="Draw network")

    draw_parser.add_argument("--msm-model-file",
                             type=str,
                             required=True,
                             help="File from which to load MSM model")
    
    draw_parser.add_argument("--figure-fl",
                             type=str,
                             required=True,
                             help="Plot filename")

    draw_parser.add_argument("--scale-size",
                             type=str,
                             choices=["observed-populations",
                                      "equilibrium-populations"],
                             default=None)

    draw_fluxes_parser = subparsers.add_parser("draw-fluxes",
                                               help="Draw fluxes")

    draw_fluxes_parser.add_argument("--msm-model-file",
                                    type=str,
                                    required=True,
                                    help="File from which to load MSM model")

    draw_fluxes_parser.add_argument("--figures-dir",
                                    type=str,
                                    required=True,
                                    help="Figures dir")

    draw_timeseries_parser = subparsers.add_parser("draw-timeseries",
                                                   help="Draw timeseries of states")

    draw_timeseries_parser.add_argument("--msm-model-file",
                                        type=str,
                                        required=True,
                                        help="File from which to load MSM model")

    draw_timeseries_parser.add_argument("--figures-dir",
                                        type=str,
                                        required=True,
                                        help="Figures dir")
    
    state_dihedral_parser = subparsers.add_parser("test-state-dihedrals",
                                                  help="Run G-tests on state by state dihedral distributions")

    state_dihedral_parser.add_argument("--msm-model-file",
                                       type=str,
                                       required=True,
                                       help="File from which to load MSM model")
    
    state_dihedral_parser.add_argument("--output-dir",
                                       type=str,
                                       required=True,
                                       help="Output directory")

    state_dihedral_parser.add_argument("--pdb-file",
                                       type=str,
                                       required=True,
                                       help="PDB file")

    state_dihedral_parser.add_argument("--input-traj",
                                       type=str,
                                       required=True,
                                       help="Input trajectory")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()

    if args.mode == "sweep-clusters":
        sweep_clusters(args)
    elif args.mode == "sweep-lag-times":
        sweep_lag_times(args)
    elif args.mode == "train-model":
        train_model(args)
    elif args.mode == "draw-network":
        plot_msm_network(args)
    elif args.mode == "draw-fluxes":
        plot_fluxes(args)
    elif args.mode == "draw-timeseries":
        plot_state_timeseries(args)
    elif args.mode == "test-state-dihedrals":
        compare_dihedral_distributions(args)
    else:
        print "Unknown mode '%s'" % args.mode
        sys.exit(1)
