import json
import os
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


def convert_energy_to_kJ_per_mol(val: float) -> float:  # Converts from kT to kJ/mol
    return N_A * val * k * 298 / 1000


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

    atom_names = list(
        permutations(list(string.ascii_lowercase) + list(string.ascii_uppercase), 2)
    )
    for i in range(0, len(positions)):
        atomname = "".join(atom_names[i])
        pos = positions[i]
        # if self_energy_calc and (i != focus_atom_index):
        #     charge = 0.0
        #     radius = 0.0
        # else:
        charge = charges[i]
        radius = radii[i]
        charge_strings.append(f"  {atomname.ljust(4)}          {charge:.4f}\n")
        radius_strings.append(f"{atomname.ljust(4)}       {float(radius):.4f}\n")
        pdb_strings.append(
            f"{'ATOM'.ljust(7)}{str(i+1).rjust(4)}  {atomname.ljust(3)}             "
            f" {str(round(pos[0],3)).rjust(8)}{str(round(pos[1],3)).rjust(8)}{str(round(pos[2],3)).rjust(8)} "
            f"                     {'Ar'.rjust(2)}\n"
        )

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
        # print(f"vacuum_energy: {vacuum_energy} kJ/mol")

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

    # Written in full by ChatGPT-4o 6/8/2024
    def calculate_born_radius_with_pure_VDW_integral(
        self, atom_index: int, scale=None, chunk_size=1000000
    ) -> float:
        """
        Take a cube containing all atoms. Then, for all gridpoints inside of the sphere circumscribed in the cube,
        use those grid points to calculate an integral dependent on whether a gridpoint is inside an atom or in solvent.
        """
        self.check()
        if not scale:
            scale = self.get_scale()

        center = np.array(self.positions[atom_index])

        # Get the diameter of the sphere in which to integrate
        d = np.max(np.linalg.norm(np.array(self.positions) - center, axis=1)) + 1 * max(
            self.radii
        )

        # Create ranges for gridpoints that are larger than the sphere
        X_range = np.arange(center[0] - d, center[0] + d, 1 / scale)
        Y_range = np.arange(center[1] - d, center[1] + d, 1 / scale)
        Z_range = np.arange(center[2] - d, center[2] + d, 1 / scale)

        def chunk_generator():
            chunk = []
            for x in X_range:
                for y in Y_range:
                    for z in Z_range:
                        point = np.array([x, y, z])
                        distance = np.linalg.norm(point - center)

                        # Check if the gridpoint is outside the focus atom but inside the sphere
                        if self.radii[atom_index] < distance < d:
                            chunk.append((point, distance))
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
            if chunk:
                yield chunk

        integral = 0.0
        num_points_in_sphere = 0
        for chunk in chunk_generator():
            points = np.array([item[0] for item in chunk])
            num_points_in_sphere += len(points)
            distances = np.array([item[1] for item in chunk])
            point_distances = np.linalg.norm(
                points[:, None, :] - np.array(self.positions), axis=2
            )
            valid_points = np.all(point_distances > self.radii, axis=1)
            integral += np.sum(1 / distances[valid_points] ** 4)

        integral /= 4 * np.pi

        # Calculate the volume of the sphere
        volume = 4 / 3 * np.pi * d**3 - 4 / 3 * np.pi * self.radii[atom_index] ** 3

        # Calculate the integral by getting the average value and multiplying by the volume
        alpha = 1 / (integral * volume / num_points_in_sphere)
        return alpha

    def get_points_on_atom_surface(
        self, atom_index: int, desired_point_density=1000
    ) -> (NDArray, int):
        """
        Return a tuple of: numpy array of xyz coordiantes of points uniformly randomly distributed
        on the surface of the sphere of atom with index `atom_index` and the number of points.
        """

        r = self.radii[atom_index]
        num_points = int(desired_point_density * 4 * np.pi * r**2)
        rng = np.random.default_rng(seed=1997)

        theta = np.arccos(2 * rng.uniform(low=0, high=1, size=num_points) - 1)
        phi = rng.uniform(low=0, high=2 * np.pi, size=num_points)
        xs = r * np.sin(theta) * np.cos(phi)
        ys = r * np.sin(theta) * np.sin(phi)
        zs = r * np.cos(theta)

        return (
            np.array([[xs[i], ys[i], zs[i]] for i in range(0, num_points)]),
            np.array([[theta[i], phi[i]] for i in range(0, num_points)]),
            num_points,
        )

    # CHEAPER BUT WRONG. HAVEN'T FIXED!! DANGER!!
    # def get_fraction_surface_points_shadowed(
    #     self, atom_index: int, desired_point_density=1000
    # ) -> float:
    #     surface_points, num_points = self.get_points_on_atom_surface(
    #         atom_index, desired_point_density
    #     )
    #     center_point = np.array(self.positions[atom_index])

    #     # Get vectors from center of atom of interest to the points on its surface
    #     surface_vecs = surface_points - center_point
    #     # surface_mags = np.linalg.norm(surface_vecs, axis=1)
    #     surface_mags = np.array(
    #         [float(self.radii[atom_index]) for _ in range(num_points)]
    #     )

    #     # Get vectors from center of atom of interest to the centers of peripheral atoms
    #     peripheral_atom_vecs = np.array(self.positions) - center_point
    #     peripheral_atom_mags = np.linalg.norm(peripheral_atom_vecs, axis=1)

    #     num_points_intersecting = 0
    #     for i, surface_vec in enumerate(surface_vecs):
    #         for j, peripheral_atom_vec in enumerate(peripheral_atom_vecs):
    #             if j == atom_index:
    #                 continue
    #             dot_product = np.dot(surface_vec, peripheral_atom_vec)
    #             theta = np.arccos(
    #                 dot_product / (surface_mags[i] * peripheral_atom_mags[j])
    #             )
    #             if theta <= np.pi/2 and abs(peripheral_atom_mags[j] * np.sin(theta)) <= self.radii[j]:
    #                 num_points_intersecting += 1
    #                 break

    #     return num_points_intersecting / num_points

    def get_fraction_surface_points_shadowed(
        self,
        atom_index: int,
        surface_point_data: tuple[NDArray, int],
    ) -> float:
        surface_points, surface_points_spherical, num_points = surface_point_data

        center_point = np.array(self.positions[atom_index])

        # Get vectors from center of atom of interest to the points on its surface
        surface_vecs = surface_points - center_point
        # surface_mags = np.linalg.norm(surface_vecs, axis=1)
        surface_mags = np.array(
            [float(self.radii[atom_index]) for _ in range(num_points)]
        )

        # Get vectors from center of atom of interest to the centers of peripheral atoms
        peripheral_atom_vecs = np.array(self.positions) - center_point
        peripheral_atom_mags = np.linalg.norm(peripheral_atom_vecs, axis=1)

        delta = 0.01
        intersection_array = []
        for i, surface_vec in enumerate(surface_vecs):
            intersection_lengths = [0 for _ in peripheral_atom_vecs]
            current_pos = surface_vec
            while np.linalg.norm(current_pos - center_point) <= 9:
                current_pos = current_pos + (surface_vec * delta)
                for j, peripheral_atom_vec in enumerate(peripheral_atom_vecs):
                    if j == atom_index:
                        continue
                    if (
                        np.linalg.norm(current_pos - peripheral_atom_vec)
                        < self.radii[j]
                    ):
                        # Calculate length of intersection path inside peripheral atom
                        # "peripheral internal vector" means the vector going from point of
                        # intersection of peripheral atom to that atom's center

                        peripheral_internal_vec = peripheral_atom_vec - current_pos

                        phi = np.arccos(
                            np.dot(surface_vec, peripheral_internal_vec)
                            / (surface_mags[i] * self.radii[j])
                        )
                        theta = 2 * phi - np.pi
                        length_of_intersection_path = np.sqrt(
                            2 * self.radii[j] ** 2
                            - 2 * np.cos(theta) * self.radii[j] ** 2
                        )
                        intersection_lengths[j] += length_of_intersection_path
                        # We already counted length due to this peripheral atom; teleport to the opposite edge
                        # of the atom to continue the ray tracing and count other lengths
                        current_pos = (
                            current_pos
                            + (length_of_intersection_path + 2 * delta) * surface_vec
                        )

            intersection_array.append(max(intersection_lengths))

        assert len(intersection_array) == num_points
        num_points_intersecting = sum([1 if v != 0 else 0 for v in intersection_array])
        return (num_points_intersecting / num_points, intersection_array)

    def get_surface_points_KL(
        self,
        atom_index: int,
        surface_point_data: tuple[NDArray, int],
        intersection_array: list[bool],
    ) -> float:
        intersection_array = np.array(intersection_array)
        surface_points, surface_points_spherical, num_points = surface_point_data
        rng = np.random.default_rng(seed=1997)
        KLs = []
        for _ in range(20):
            uniform_random_reference = rng.permutation(intersection_array)
            KL = np.sum(
                (intersection_array + 1)
                * np.log((intersection_array + 1) / (uniform_random_reference + 1))
            )
            KLs.append(KL)
        return KLs


def get_surface_point_info(
    system: System, atom_index: int, desired_point_density=1000
) -> tuple:
    surface_point_data = system.get_points_on_atom_surface(
        atom_index, desired_point_density
    )
    fraction_shadowed, intersection_array = system.get_fraction_surface_points_shadowed(
        atom_index, surface_point_data
    )
    KLs = system.get_surface_points_KL(
        atom_index, surface_point_data, intersection_array
    )
    return surface_point_data, intersection_array, fraction_shadowed, KLs


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

    result_born_radius = get_born_radius(system, 0)
    # result_born_radius = (None, None)
    result_surface = get_surface_point_info(system, 0, 1000)
    result_dict = {
        "system": system,
        "condor_process": int(sys.argv[1]),
        "values": {
            "born_radius_with_pb": result_born_radius[0],
            "born_radius_with_vdw_integral": result_born_radius[1],
            "fraction_surface_points_shadowed": result_surface[2],
            "KL": sum(result_surface[3]) / len(result_surface[3]),
        },
    }
    with open("result.pickle", "wb+") as f:
        pickle.dump(result_dict, f)
    print(str(result_dict))
