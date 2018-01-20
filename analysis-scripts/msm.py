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
import numpy as np
from sklearn.cluster import k_means
from sklearn.externals import joblib
import numpy.linalg as LA

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

        self.obs_pop_counts = np.zeros(self.n_states,
                                  dtype=np.int)
        for idx in self.labels:
            self.obs_pop_counts[idx] += 1

        counts = np.zeros((self.n_states,
                           self.n_states))

        for i, from_ in enumerate(self.labels):
            j = i + self.stride
            if j < len(self.labels):
                to_ = self.labels[j]
                counts[to_, from_] += 1

        # for prettier printing
        print counts.astype(np.int32)

        # force symmetry
        self.sym_counts = 0.5 * (counts + counts.T)

        # normalize columns
        self.transitions = self.sym_counts / self.sym_counts.sum(axis=1)[:, None]

        # get right eigenvectors
        u, v = LA.eig(self.transitions)

        # re-order in descending order
        sorted_idx = np.argsort(u)[::-1]    
        u = u[sorted_idx]
        v = v[:, sorted_idx]

        self.timescales = - self.timestep * self.stride / np.log(u[1:])
        self.equilibrium_dist = v[:, 0] / v[:, 0].sum()
        

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
    n_timescales = timescales.shape[0]
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
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()

    if args.mode == "sweep-clusters":
        sweep_clusters(args)
    elif args.mode == "sweep-lag-times":
        sweep_lag_times(args)
    elif args.mode == "train-model":
        train_model(args)
    else:
        print "Unknown mode '%s'" % args.mode
        sys.exit(1)
