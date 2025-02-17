import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import pickle
import random
import string
import subprocess
import sys
import time
from copy import copy
from datetime import datetime
from itertools import permutations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import N_A, elementary_charge, epsilon_0, k, pi

ListOfVals = ArrayLike
ListOfListsOfVals = ArrayLike


def born(energy: float) -> float:  # Assumes given in kJ/mol
    energy *= 1000  # Convert from kJ/mol to J/mol
    radius = (
        -(N_A / energy) * (elementary_charge**2 / (8 * pi * epsilon_0)) * (1 - 1 / 80)
    )  # in meters
    radius *= 1e10  # Change from meters to Angstrom
    return radius


def convert_energy_to_kJ_per_mol(val: float) -> float:  # Converts from kT/e to kJ/mol
    return N_A * val * k * 298 / 1000


def convert_potential_to_V(val: float) -> float:  # Converts from kT/e to V
    conversion_factor = (
        k * 298
    ) / elementary_charge  # This is how many volts are in one kT/e
    return val * conversion_factor


def get_distance(
    position1: ListOfVals, position2: ListOfVals
) -> float:  # Gets the distance between two lists of three floats
    return np.sqrt(np.sum([(position1[i] - position2[i]) ** 2 for i in range(0, 3)]))


def convert_to_polar(position: ListOfVals, center: ListOfVals) -> ListOfVals:
    """
    Takes in a cartesian position and returns tuple (r, theta, phi).
    """
    rel_x, rel_y, rel_z = [position[i] - center[i] for i in range(3)]
    r = get_distance(position, center)
    theta = np.arccos(rel_z / r)
    phi = np.sign(rel_y) * np.arccos(rel_x / np.sqrt(rel_x**2 + rel_y**2))
    return (r, theta, phi)


def run_pb(
    positions, radii, charges, focus_atom_index=None, cpus=1, scale=10, indi=1, exdi=80
) -> dict:
    """
    Runs Poisson-Boltzmann on working_data/params.{id}.prm and outputs to working_data/output.log
    """

    # positions, radii, charges, self_energy_calc, focus_atom_index, cpus = input

    assert (
        len(positions) == len(radii) == len(charges)
    ), "Lengths of positions, radii, and charges must be identical!"

    # assert not (self_energy_calc ^ (focus_atom_index is not None)), (
    #     "self_energy_calc and focus_atom_index must be both unspecified or both"
    #     " specified!"
    # )

    rng = np.random.default_rng()
    id = "".join(random.choices(string.ascii_letters + string.digits, k=16))

    charge_strings = ["atom__resnumbc_charge_\n"]
    radius_strings = ["atom__res_radius_\n"]
    pdb_strings = []

    atom_names = list(permutations(list(string.ascii_uppercase), 2))
    for i in range(0, len(positions)):
        atomname = "".join(atom_names[i])
        pos = positions[i]
        # if self_energy_calc and (i != focus_atom_index):
        #     charge = 0.0
        #     radius = 0.0
        # else:
        charge = charges[i]
        radius = radii[i]
        charge_strings.append(f"  {atomname.ljust(4)}          {float(charge):.4f}\n")
        radius_strings.append(f"{atomname.ljust(4)}       {float(radius):.4f}\n")
        pdb_strings.append(
            f"{'ATOM'.ljust(7)}{str(i+1).rjust(4)}  {atomname.ljust(3)}             "
            f" {str(round(pos[0],3)).rjust(8)}{str(round(pos[1],3)).rjust(8)}{str(round(pos[2],3)).rjust(8)} "
            f"                     {'Ar'.rjust(2)}\n"
        )

    if not os.path.exists("working_data"):
        os.makedirs("working_data")
    with open(f"working_data/custom.{id}.crg", "w") as f:
        f.writelines(charge_strings)
    with open(f"working_data/custom.{id}.siz", "w") as f:
        f.writelines(radius_strings)
    with open(f"working_data/custom.{id}.pdb", "w") as f:
        f.writelines(pdb_strings)

    lines = []
    with open("params.prm", "r") as f:
        for line in f.readlines():
            l = line.replace("%%SCALE%%", str(scale))
            l = l.replace("%%ID%%", id)
            if focus_atom_index is not None:
                cx, cy, cz = positions[focus_atom_index]
                center = f"{cx:0.4f},{cy:0.4f},{cz:0.4f}"
            else:
                center = "0.0,0.0,0.0"
            l = l.replace("%%CENTER%%", center)
            # if ("indi=" in line) and (not line[0] == "!"):
            #     indi = float(
            #         "".join([c if c.isnumeric() or c == "." else "" for c in line])
            #     )
            # if ("exdi=" in line) and (not line[0] == "!"):
            #     exdi = float(
            #         "".join([c if c.isnumeric() or c == "." else "" for c in line])
            #     )
            l = l.replace("%%INDI%%", str(indi))
            l = l.replace("%%EXDI%%", str(exdi))
            lines.append(l)
    with open(f"working_data/params.{id}.prm", "w") as f:
        f.writelines(lines)

    # os.system(f"mpirun -np {np} delphi params.prm.{id} > output.log.{id}")
    # subprocess.run([f"mpirun -np {cpus} delphi params.prm.{id} > output.log.{id}".split(" ")], stdout=subprocess.PIPE).stdout.decode('utf-8')
    try:
        output = subprocess.check_output(
            [
                (
                    (
                        f"mpirun -np {cpus} {os.environ['DELPHIEXEC']} working_data/params.{id}.prm >"
                        f" working_data/output.log"
                    )
                    if (cpus != 1)
                    else (
                        f"{os.environ['DELPHIEXEC']} working_data/params.{id}.prm >"
                        f" working_data/output.log"
                    )
                ),
                f"sleep 5s",
            ],
            shell=True,
            text=True,
        )
    except:
        print("ERROR RUNNING DELPHI!")
        return {"total_grid": rng.integers(0, 1000)}
    with open(f"working_data/output.log", "r") as f:
        energies = {}
        counter = 0
        for line in f.readlines()[::-1]:
            if "Energy> All required energy terms but grid energy" in line:
                energies["all_but_grid"] = float(line[:69][-10:])
                counter += 1
            if "Energy> Coulombic energy" in line:
                energies["coulombic"] = float(line[:69][-10:])
                counter += 1
            if "Energy> Corrected reaction field energy" in line:
                energies["corrected_RF"] = float(line[:69][-10:])
                counter += 1
            if "Energy> Total grid energy" in line:
                energies["total_grid"] = float(line[:69][-10:])
                counter += 1
            if counter >= 4:
                break

    energies["grid_pots_at_atoms"] = {}
    energies["coul_pots_at_atoms"] = {}
    with open(f"working_data/potentials.{id}.frc", "r") as f:
        should_read = False
        for line in f.readlines():
            if (
                "ATOM DESCRIPTOR         ATOM COORDINATES (X,Y,Z)    GRID PT. COUL. POT"
                in line
            ):
                should_read = True
                continue
            if should_read and (not "total energy =" in line):
                energies["grid_pots_at_atoms"][
                    atom_names.index(tuple(line[0:2]))
                ] = float(line[50:60])
                energies["coul_pots_at_atoms"][
                    atom_names.index(tuple(line[0:2]))
                ] = float(line[60:70])

    os.remove(f"working_data/params.{id}.prm")
    os.remove(f"working_data/custom.{id}.crg")
    os.remove(f"working_data/custom.{id}.siz")
    os.remove(f"working_data/custom.{id}.pdb")
    os.remove(f"working_data/output.log")
    os.remove(f"working_data/potentials.{id}.frc")
    # os.remove(f"working_data/phimap.{id}.cube")

    return energies


ListOfVals = list[float]
ListOfListsOfVals = list[list[float]]


class System:
    """
    A system of atoms.
    """

    def __init__(self, positions=None, radii=None, charges=None) -> None:
        self.positions, self.radii, self.charges = (None, None, None)

        if positions is not None:
            self.set_positions(positions)
        if radii is not None:
            self.set_radii(radii)
        if charges is not None:
            self.set_charges(charges)

        self.graph = None
        self.id = "".join(random.choices(string.ascii_letters + string.digits, k=16))

    def set_positions(self, positions: ListOfListsOfVals, overwrite=False) -> None:
        if self.positions is not None and not overwrite:
            raise ValueError(
                "self.positions is already set. To overwrite, specify overwrite=True"
            )
        else:
            self.positions = positions

    def set_radii(self, radii: ListOfVals, overwrite=False) -> None:
        if self.radii is not None and not overwrite:
            raise ValueError(
                "self.radii is already set. To overwrite, specify overwrite=True"
            )
        else:
            self.radii = radii

    def set_charges(self, charges: ListOfVals, overwrite=False) -> None:
        if self.charges is not None and not overwrite:
            raise ValueError(
                "self.charges is already set. To overwrite, specify overwrite=True"
            )
        else:
            self.charges = charges

    def check(self) -> None:
        checklist = [
            attribute is None
            for attribute in (self.positions, self.radii, self.charges)
        ]
        if True in checklist:
            empty_attribute = ("positions", "radii", "charges")[checklist.index(True)]
            raise ValueError(f"The attribute self.{empty_attribute} is None")

        checklist = [
            isinstance(attribute, (list, tuple, np.ndarray))
            for attribute in (self.positions, self.radii, self.charges)
        ]
        if False in checklist:
            mistyped_attribute = ("positions", "radii", "charges")[
                checklist.index(False)
            ]
            raise ValueError(
                f"The attribute self.{mistyped_attribute} is not of type list or tuple."
            )

        if not len(self.positions) == len(self.radii) == len(self.charges):
            raise ValueError(
                "The attributes self.positions, self.radii, and self.charges are not"
                " the same length"
                f" ({[len(self.positions), len(self.radii), len(self.charges)]})"
            )

    def get_scale(self) -> float:
        """
        Return the DelPhi `scale` parameter that should be used. Higher values make for
        more accurate but more time/memory hungry calculations. Scales cubically!
        """

        # While we do these larger grids, ensure the grid scale is 10
        return 10

        if self.radii is None:
            raise ValueError(
                "At least one radius must be defined to use self.get_scale()"
            )
        count = 0
        for radius in self.radii:
            if radius > 0:
                count += 1
        if count <= 3:
            return 30
        else:
            return 10

    def calculate_born_radius_with_pb(self, atom_index: int) -> float:
        """
        Finds the solvation free energy of the atom with index `atom_index` via PB, then uses the Born equation
        to determine what radius of an ion of the same charge would have to be to recreate the same solvation
        free energy in a system with no other atoms.
        """

        # Step 1: Get the energy of the ion in vacuum (with all other atoms neutral)

        # Step 2: Get the energy of the ion in water (with all other atoms neutral)

        # Step 3: Use the difference in energies (the solvation free energy) in the Born equation

        self.check()
        scale = self.get_scale()

        temp_charges = []
        for i, charge in enumerate(self.charges):
            if i == atom_index:
                temp_charges.append(charge)
            else:
                temp_charges.append(0)

        vacuum_energies = run_pb(
            positions=self.positions,
            radii=self.radii,
            charges=temp_charges,
            focus_atom_index=atom_index,
            cpus=1,
            scale=scale,
            indi=1,
            exdi=1,
        )
        vacuum_energy = convert_energy_to_kJ_per_mol(vacuum_energies["total_grid"])
        print(f"vacuum_energy: {vacuum_energy} kJ/mol")

        solvated_energies = run_pb(
            positions=self.positions,
            radii=self.radii,
            charges=temp_charges,
            focus_atom_index=atom_index,
            cpus=1,
            scale=scale,
            indi=1,
            exdi=80,
        )
        solvated_energy = convert_energy_to_kJ_per_mol(solvated_energies["total_grid"])
        # print(f"solvated_energy: {solvated_energy} kJ/mol")

        solvation_free_energy = solvated_energy - vacuum_energy
        # print(f"solvation_free_energy: {solvation_free_energy} kJ/mol")

        return born(solvation_free_energy)

    def calculate_born_radius_with_pure_VDW_integral_old(
        self, atom_index: int, scale=None
    ) -> float:
        self.check()
        if not scale:
            scale = self.get_scale()

        # Calcualte the center of mass of the atoms in the system
        # center_of_mass = [
        #     sum([self.positions[j][i] for j in range(len(self.positions))])
        #     / len(self.positions)
        #     for i in range(3)
        # ]
        center = np.array(self.positions[atom_index])

        # Get the diameter of the sphere in which to integrate
        d = np.max(np.linalg.norm(np.array(self.positions) - center, axis=1)) + 1 * max(
            self.radii
        )

        # Create a cube of gridpoints that is larger than the sphere
        X = np.arange(center[0] - d, center[0] + d, 1 / scale)
        Y = np.arange(center[1] - d, center[1] + d, 1 / scale)
        Z = np.arange(center[2] - d, center[2] + d, 1 / scale)
        X, Y, Z = np.meshgrid(X, Y, Z)

        points_in_cube = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
        distances = np.linalg.norm(points_in_cube - center, axis=1)
        points_in_sphere = points_in_cube[
            (self.radii[atom_index] < distances) & (distances < d)
        ]

        # Keep a list of gridpoints that are inside the sphere in which to integrate
        # for point in points_in_sphere:
        #     in_solvent = True
        #     # Check if this point is inside an atom
        #     for i, position in enumerate(self.positions):
        #         if get_distance(point, position) < self.radii[i]:
        #             in_solvent = False
        #             break
        #     if in_solvent:
        #         r, theta, phi = convert_to_polar(point, center)
        #         integral += 1 / (4 * np.pi * (r**4))

        integral = np.sum(
            np.all(
                np.linalg.norm(
                    points_in_sphere[:, None, :] - np.array(self.positions), axis=2
                )
                > self.radii,
                axis=1,
            )
            / np.linalg.norm(points_in_sphere - center, axis=1) ** 4
        ) / (4 * np.pi)

        # Calculate the volume of the sphere
        volume = 4 / 3 * np.pi * d**3 - 4 / 3 * np.pi * self.radii[atom_index] ** 3

        # Calculate the integral by getting the average value and multiplying by the volume
        alpha = 1 / (integral * volume / len(points_in_sphere))
        return alpha

    def calculate_interatomic_potential(self, atom_indices: tuple[int]):
        """
        Use PB calculations to get the potential on atom 2 due to atom 1's charge.
        """
        self.check()
        scale = self.get_scale()
        atom_index1, atom_index2 = atom_indices

        # Make sure that all atoms are neutral, excluding atom 1
        # print(self.charges)
        assert not np.any(self.charges[1:] != 0)

        # Get potential at atom 2
        energies = run_pb(
            self.positions,
            self.radii,
            self.charges,
            focus_atom_index=atom_index1,  # Focus atom index isn't where we're getting the potential necessarily
            scale=scale,
            indi=1,
            exdi=80,
        )
        interatomic_potential = convert_potential_to_V(
            energies["grid_pots_at_atoms"][atom_index2]
        )

        return interatomic_potential

    # Fully rewritten for optimization by ChatGPT-4o 6/8/2024
    def calculate_born_radius_with_pure_VDW_integral(
        self, atom_index: int, scale=None, chunk_size=1000000
    ) -> float:
        """
        Calculate the Born radius using pure VDW integral. This is an optimized version
        of the original function to improve speed and memory efficiency.
        """
        self.check()
        if not scale:
            scale = self.get_scale()

        center = np.array(self.positions[atom_index])

        # Get the diameter of the sphere in which to integrate
        D = np.max(np.linalg.norm(np.array(self.positions) - center, axis=1)) + max(
            self.radii
        )

        # Create ranges for grid points
        grid_range = np.arange(-D, D, 1 / scale)
        grid_points = np.stack(
            np.meshgrid(grid_range, grid_range, grid_range), -1
        ).reshape(-1, 3)

        # Pre-calculate radii squares for quick checks
        radii_squared = np.array(self.radii) ** 2

        def chunk_generator():
            for start in range(0, grid_points.shape[0], chunk_size):
                end = min(start + chunk_size, grid_points.shape[0])
                chunk = grid_points[start:end]
                distances = np.linalg.norm(chunk, axis=1)

                # Mask for points within the spherical shell (excluding inside atom)
                mask = (self.radii[atom_index] < distances) & (distances < D)
                yield chunk[mask] + center, distances[mask]

        integral = 0.0
        num_points_in_sphere = 0

        for points, distances_from_center in chunk_generator():
            num_points_in_sphere += len(points)

            # Determine which points are in solvent
            in_solvent_mask = np.ones(len(points), dtype=bool)

            for i, position in enumerate(self.positions):
                distance_to_position = np.linalg.norm(points - position, axis=1)
                in_solvent_mask &= distance_to_position >= self.radii[i]

            # Filter points that are in solvent
            solvent_distances = distances_from_center[in_solvent_mask]

            if len(solvent_distances) > 0:
                integral += np.sum(1 / solvent_distances**4)

        integral /= 4 * np.pi

        # Calculate the volume of the sphere
        volume = 4 / 3 * np.pi * (D**3 - self.radii[atom_index] ** 3)

        # Calculate the integral by getting the average value and multiplying by the volume
        alpha = 1 / (integral * volume / num_points_in_sphere)
        return alpha

    def calculate_born_radius_with_pure_VDW_integral_hemisphere(
        self, atom_indices=(0, 1), scale=None, chunk_size=1000000
    ) -> float:
        """
        Calculate the Born radius using pure VDW integral over a hemisphere.
        This function is optimized for speed and memory efficiency.
        """
        self.check()
        if not scale:
            scale = self.get_scale()

        atom1, atom2 = atom_indices
        center = np.array(self.positions[atom1])
        partner = np.array(self.positions[atom2])

        # Get the diameter of the sphere in which to integrate
        D = np.max(np.linalg.norm(np.array(self.positions) - center, axis=1)) + max(
            self.radii
        )

        # Create ranges for grid points
        grid_range = np.arange(-D, D, 1 / scale)
        grid_points = np.stack(
            np.meshgrid(grid_range, grid_range, grid_range), -1
        ).reshape(-1, 3)

        # Vector for hemisphere determination
        partner_vector = partner - center
        partner_distance = np.linalg.norm(partner_vector)
        partner_unit_vector = partner_vector / partner_distance

        # Pre-calculate radii squares for quick checks
        radii_squared = np.array(self.radii) ** 2

        def chunk_generator():
            for start in range(0, grid_points.shape[0], chunk_size):
                end = min(start + chunk_size, grid_points.shape[0])
                chunk = grid_points[start:end]

                distances = np.linalg.norm(chunk, axis=1)
                direction_vectors = chunk / distances[:, np.newaxis]

                # Mask for points within the spherical shell and in the desired hemisphere
                hemisphere_mask = np.dot(direction_vectors, partner_unit_vector) >= 0
                shell_mask = (self.radii[atom1] < distances) & (distances < D)

                mask = hemisphere_mask & shell_mask
                yield chunk[mask] + center, distances[mask]

        integral = 0.0
        num_points_in_hemisphere = 0

        for points, distances_from_center in chunk_generator():
            num_points_in_hemisphere += len(points)

            # Determine which points are in solvent
            in_solvent_mask = np.ones(len(points), dtype=bool)

            for i, position in enumerate(self.positions):
                distance_to_position = np.linalg.norm(points - position, axis=1)
                in_solvent_mask &= distance_to_position >= self.radii[i]

            # Filter points that are in solvent
            solvent_distances = distances_from_center[in_solvent_mask]

            if len(solvent_distances) > 0:
                integral += np.sum(1 / solvent_distances**4)

        integral /= 4 * np.pi

        # Account for the other hemisphere... assume it's the same!
        integral *= 2
        num_points_in_sphere = 2 * num_points_in_hemisphere

        # Calculate the volume of the sphere (not hemisphere, arbitrary factory of 2 difference here)
        volume = 4 / 3 * np.pi * (D**3 - self.radii[atom1] ** 3)

        # Calculate the integral by getting the average value and multiplying by the volume
        alpha = 1 / (integral * volume / num_points_in_sphere)
        return alpha[0]

    def calculate_born_radius_with_pure_VDW_integral_by_octants(
        self, atom_index: int, scale=None, chunk_size=1000000
    ) -> list:
        """
        Calculate the Born radius using pure VDW integral, returning one radius for each octant.
        This is an optimized version to improve speed and memory efficiency.
        """
        self.check()
        if not scale:
            scale = self.get_scale()

        center = np.array(self.positions[atom_index])

        # Get the diameter of the sphere in which to integrate
        D = np.max(np.linalg.norm(np.array(self.positions) - center, axis=1)) + max(
            self.radii
        )

        # Create ranges for grid points
        grid_range = np.arange(-D, D, 1 / scale)
        grid_points = np.stack(
            np.meshgrid(grid_range, grid_range, grid_range), -1
        ).reshape(-1, 3)

        # Pre-calculate radii squares for quick checks
        radii_squared = np.array(self.radii) ** 2

        # Initialize accumulators for each octant
        integrals = np.zeros(8)
        num_points_in_sphere = np.zeros(8)

        def chunk_generator():
            for start in range(0, grid_points.shape[0], chunk_size):
                end = min(start + chunk_size, grid_points.shape[0])
                chunk = grid_points[start:end]
                distances = np.linalg.norm(chunk, axis=1)

                # Mask for points within the spherical shell (excluding inside atom)
                mask = (self.radii[atom_index] < distances) & (distances < D)
                yield chunk[mask] + center, distances[mask]

        def get_octant_index(point):
            """Determine which octant a point belongs to based on its (x, y, z) coordinates."""
            return (
                (point[0] >= center[0]) << 2
                | (point[1] >= center[1]) << 1
                | (point[2] >= center[2])
            )

        for points, distances_from_center in chunk_generator():
            # Determine which points are in solvent
            in_solvent_mask = np.ones(len(points), dtype=bool)

            for i, position in enumerate(self.positions):
                distance_to_position = np.linalg.norm(points - position, axis=1)
                in_solvent_mask &= distance_to_position >= self.radii[i]

            # Filter points that are in solvent
            solvent_points = points[in_solvent_mask]
            solvent_distances = distances_from_center[in_solvent_mask]

            if len(solvent_points) > 0:
                # Compute integrals for each octant
                for i, point in enumerate(solvent_points):
                    octant_index = get_octant_index(point)
                    integrals[octant_index] += 1 / solvent_distances[i] ** 4
                    num_points_in_sphere[octant_index] += 1

        # Normalize integrals and calculate alpha for each octant
        octant_alphas = []
        for octant in range(8):
            if num_points_in_sphere[octant] > 0:
                integrals[octant] /= 4 * np.pi
                # Calculate the volume of the sphere for the octant
                volume = (
                    (4 / 3) * np.pi * (D**3 - self.radii[atom_index] ** 3)
                )  # We're suggesting that the entire sphere is made up of eight clones of this octant, so we don't divide by 8 in the volume calculation
                # Calculate alpha (Born radius) for the octant
                alpha = 1 / (integrals[octant] * volume / num_points_in_sphere[octant])
                octant_alphas.append(alpha)
            else:
                octant_alphas.append(float("inf"))  # Handle empty octants

        return octant_alphas

    def create_networkx_graph(self, delete_existing=False):
        """
        Create a complete graph in networkx where nodes are atoms and edges
        are interatomic distances. Nodes can be given properties using
        methods like `self.assign_radii_to_graph()`.
        """
        assert (
            self.graph is None
        ) or delete_existing, "A graph already exists for this `System`. To rewrite it, use `delete_existing=True`."
        graph = nx.Graph()

        for i, position in enumerate(system.positions):
            graph.add_node(i)
        for a, b in combinations(range(0, len(system.positions)), r=2):
            dist = np.linalg.norm(
                np.array(system.positions[a]) - np.array(system.positions[b])
            )
            graph.add_edge(a, b, weight=dist)

        self.graph = graph
        return

    def assign_radii_to_graph(self):
        """
        Give each node in `self.graph` the property of its cavity radius.
        """
        assert (
            self.graph is not None
        ), "The `System` must have a graph before calling `self.assign_radii_to_graph()`."
        radii_dict = {}
        for i, node in enumerate(self.graph.nodes):
            radii_dict[i] = {"radius": self.radii[i]}
        nx.set_node_attributes(self.graph, radii_dict)
        return


def get_born_radius(system: System, atom_index: int) -> tuple:
    BR_pb = system.calculate_born_radius_with_pb(atom_index)
    BR_pure_VDW = system.calculate_born_radius_with_pure_VDW_integral(atom_index)
    return (BR_pb, BR_pure_VDW)


if __name__ == "__main__":
    with open("system_input.pickle", "rb") as f:
        system = pickle.load(f)
    # system_input = np.reshape(system_input, (int(np.shape(system_input)[0] / 7), 7))

    # radii, charges, positions = [], [], []
    # system = System()
    # for i, atom_input in enumerate(system_input):
    #     r, d, theta, phi, x, y, z = atom_input
    #     radii.append(float(r))
    #     charges.append(1)
    #     positions.append([float(x), float(y), float(z)])
    # system.set_radii(radii)
    # system.set_charges(charges)
    # system.set_positions(positions)

    interaction_indices = (0, 1)
    result_potential = system.calculate_interatomic_potential(interaction_indices)
    BR_pb = system.calculate_born_radius_with_pb(atom_index=0)
    BR_pure_VDW1 = system.calculate_born_radius_with_pure_VDW_integral(atom_index=0)
    BR_pure_VDW2 = system.calculate_born_radius_with_pure_VDW_integral(atom_index=1)

    # BR_pure_VDW_hemisphere = (
    #     system.calculate_born_radius_with_pure_VDW_integral_hemisphere(
    #         atom_indices=interaction_indices
    #     )
    # )
    BR_pure_VDW_by_octants1 = (
        system.calculate_born_radius_with_pure_VDW_integral_by_octants(atom_index=0)
    )
    BR_pure_VDW_by_octants2 = (
        system.calculate_born_radius_with_pure_VDW_integral_by_octants(atom_index=1)
    )

    system.values = {
        "interaction_potential": (interaction_indices, result_potential),
        "BR_pb": (BR_pb, None),
        "BR_pure_VDW": (BR_pure_VDW1, BR_pure_VDW2),
        "BR_pure_VDW_by_octants": (BR_pure_VDW_by_octants1, BR_pure_VDW_by_octants2),
    }

    with open("result.pickle", "wb+") as f:
        pickle.dump(system, f)
    print(str(system.values))
