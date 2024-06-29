"""
ray_based/prepare.py

This file generates `N` systems each with `n` atoms.
Each system has one central atom with a randomly-populated
environment surrounding it.
"""

import os
import pickle
from itertools import combinations

import numpy as np

from worker import System

rng = np.random.default_rng(seed=1997)

max_atom_count = 10
n_systems_per_atom_count = 2
total_count = n_systems_per_atom_count * (max_atom_count - 1)


def generate_random_values(ranges, size):
    return np.array([rng.uniform(r[0], r[1], size=size) for r in ranges]).T


def check_overlap(xs, ys, zs, rs):
    xs, ys, zs, rs = np.array(xs), np.array(ys), np.array(zs), np.array(rs)
    indices = np.arange(len(xs))
    for i, j in combinations(indices, 2):
        distance = np.sqrt(
            (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 + (zs[i] - zs[j]) ** 2
        )
        if distance < (rs[i] + rs[j]):
            return True
    return False


if __name__ == "__main__":
    inputs = []
    for atom_count in range(2, max_atom_count + 1):
        print(f"Generating {n_systems_per_atom_count} {atom_count}-atom systems...")
        radii_ranges = [(0.8, 2.0)] * atom_count
        distance_ranges = [(0, 0)] + [(max(r[1] for r in radii_ranges), 10)] * (
            atom_count - 1
        )
        theta_ranges = [(0, 0)] * 2 + [(0, np.pi)] * (atom_count - 2)
        phi_ranges = [(0, 2 * np.pi)] * atom_count

        attempts, successful_attempts = 0, 0
        while successful_attempts < n_systems_per_atom_count:
            attempts += 1

            rs = generate_random_values(radii_ranges, 1).flatten()
            ds = generate_random_values(distance_ranges, 1).flatten()
            thetas = generate_random_values(theta_ranges, 1).flatten()
            phis = generate_random_values(phi_ranges, 1).flatten()

            xs = ds * np.sin(thetas) * np.cos(phis)
            ys = ds * np.sin(thetas) * np.sin(phis)
            zs = ds * np.cos(thetas)

            if check_overlap(xs, ys, zs, rs):
                continue

            system = System()
            system.set_radii(rs)
            system.set_charges([1.0] * atom_count)
            system.set_positions(np.column_stack((xs, ys, zs)).tolist())
            inputs.append(system)
            successful_attempts += 1

        print(
            f"It took {attempts} attempts to create {n_systems_per_atom_count} systems.\n"
        )

    for i, system in enumerate(inputs):
        os.makedirs(f"results/{i}", exist_ok=True)
        os.makedirs(f"worker_logs/{i}", exist_ok=True)
        with open(f"worker_logs/{i}/system_input.pickle", "wb") as f:
            pickle.dump(system, f)

    print(f"Total count: {total_count} systems")
