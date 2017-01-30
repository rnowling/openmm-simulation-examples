"""
Script for simulating models with the AMBER forcefields and GBSA implicit solvent.
"""

import argparse
from sys import stdout

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

def get_forcefield(forcefield_name):
    if "amber96" in forcefield_name:
        gbsa_fl = "amber96_obc.xml"
    elif "amber99" in forcefield_name:
        gbsa_fl = "amber99_obc.xml"
    elif "amber03" in forcefield_name:
        gbsa_fl = "amber03_obc.xml"
    elif "amber10" in forcefield_name:
        gbsa_fl = "amber10_obc.xml"
    else:
        raise Exception, "No GBSA parameters available for forcefield '%s'" % forcefield_name

    # These files are provided as part of OpenMM
    # We specify the usage of GBSA implicit solvent by
    # including the amber*_obc.xml files
    # Order is important -- need to load regular forcefield
    # file before GBSA file
    return ForceField(forcefield_name + ".xml", gbsa_fl)


def run_simulation(n_steps, cutoff_distance, temperature, damping, forcefield_name, positions_fl, energies_fl, output_steps):
    pdb = PDBFile("models/alanine-dipeptide-implicit.pdb")

    forcefield = get_forcefield(forcefield_name)

    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=CutoffNonPeriodic,
                                     nonbondedCutoff=cutoff_distance * nanometer)

    integrator = LangevinIntegrator(temperature * kelvin,
                                    damping / picosecond,
                                    0.001 * picosecond)

    simulation = Simulation(pdb.topology,
                            system,
                            integrator)

    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()


    simulation.reporters.append(DCDReporter(positions_fl, output_steps))

    simulation.reporters.append(StateDataReporter(stdout,
                                                  output_steps,
                                                  time=True,
                                                  step=True,
                                                  kineticEnergy=True,
                                                  potentialEnergy=True,
                                                  totalEnergy=True,
                                                  temperature=True))

    simulation.reporters.append(StateDataReporter(energies_fl,
                                                  output_steps,
                                                  time=True,
                                                  step=True,
                                                  kineticEnergy=True,
                                                  potentialEnergy=True,
                                                  totalEnergy=True,
                                                  temperature=True))
    
    simulation.step(n_steps)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps",
                        type=int,
                        required=True,
                        help="Number of steps (each step is 1 fs)")
    
    parser.add_argument("--temperature",
                        type=float,
                        default=300.,
                        help="Temperature (in kelvin, default 300)")

    parser.add_argument("--damping",
                        type=float,
                        default=91.,
                        help="Langevin damping factor (in ps^-1, default 91/ps)")

    parser.add_argument("--cutoff-dist",
                        type=float,
                        default=1.,
                        help="Cutoff distance (in nm, default 10 nm)")

    parser.add_argument("--positions-fl",
                        type=str,
                        required=True,
                        help="File to write positions out to (Needs .pdb or .dcd extension)")

    parser.add_argument("--energies-fl",
                        type=str,
                        required=True,
                        help="File to write energies out to")

    parser.add_argument("--steps-output",
                        type=int,
                        default=10000,
                        help="Period for writing out energies and positions (in steps)")

    parser.add_argument("--forcefield",
                        type=str,
                        choices=["amber96",
                                 "amber99sb",
                                 "amber99sbildn",
                                 "amber99sbnmr",
                                 "amber03",
                                 "amber10"],
                        default="amber96",
                        help="Choice of Amber forcefield")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    run_simulation(args.steps,
                   args.cutoff_dist,
                   args.temperature,
                   args.damping,
                   args.forcefield,
                   args.positions_fl,
                   args.energies_fl,
                   args.steps_output)

    
