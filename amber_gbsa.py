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

def run_simulation(pdb_file, n_steps, cutoff_distance, timestep, temperature, damping, forcefield_name, positions_fl, energies_fl, output_steps, minimize, input_state, output_state):
    pdb = PDBFile(pdb_file)

    forcefield = get_forcefield(forcefield_name)

    positions = pdb.getPositions()
    topology = pdb.getTopology()

    system = forcefield.createSystem(topology,
                                     nonbondedMethod=CutoffNonPeriodic,
                                     nonbondedCutoff=cutoff_distance * nanometer)
    integrator = LangevinIntegrator(temperature * kelvin,
                                    damping / picosecond,
                                    timestep * picosecond)

    simulation = Simulation(topology,
                            system,
                            integrator,
                            state=input_state)

    platform = simulation.context.getPlatform()
    print "Platform:", platform.getName(), platform.supportsDoublePrecision()
    print "System:", system.usesPeriodicBoundaryConditions()

    if not input_state:
        simulation.context.setPositions(positions)

    if minimize:
        simulation.minimizeEnergy()

    simulation.reporters.append(DCDReporter(positions_fl, output_steps))

    simulation.reporters.append(StateDataReporter(stdout,
                                                  output_steps,
                                                  time=True,
                                                  step=True,
                                                  kineticEnergy=True,
                                                  potentialEnergy=True,
                                                  totalEnergy=True,
                                                  temperature=True,
                                                  speed=True))

    simulation.reporters.append(StateDataReporter(energies_fl,
                                                  output_steps,
                                                  time=True,
                                                  step=True,
                                                  kineticEnergy=True,
                                                  potentialEnergy=True,
                                                  totalEnergy=True,
                                                  temperature=True))
    simulation.step(n_steps)
    if output_state:
        simulation.saveState(output_state)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps",
                        type=int,
                        required=True,
                        help="Number of steps")
    
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

    parser.add_argument("--minimize",
                        action="store_true",
                        help="Minimize before running simulation")

    parser.add_argument("--steps-output",
                        type=int,
                        default=10000,
                        help="Period for writing out energies and positions (in steps)")

    parser.add_argument("--timestep",
                        type=float,
                        default=0.001,
                        help="Timesteps in ps")

    parser.add_argument("--forcefield",
                        type=str,
                        choices=["amber96",
                                 "amber99sb",
                                 "amber99sbildn",
                                 "amber99sbnmr",
                                 "amber03",
                                 "amber10"],
                        required=True,
                        help="Choice of Amber forcefield")

    parser.add_argument("--pdb-file",
                        type=str,
                        required=True,
                        help="Structure file")

    parser.add_argument("--input-state",
                        type=str,
                        help="Structure file")

    parser.add_argument("--output-state",
                        type=str,
                        help="Structure file")


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    run_simulation(args.pdb_file,
                   args.steps,
                   args.cutoff_dist,
                   args.timestep,
                   args.temperature,
                   args.damping,
                   args.forcefield,
                   args.positions_fl,
                   args.energies_fl,
                   args.steps_output,
                   args.minimize,
                   args.input_state,
                   args.output_state)

    
