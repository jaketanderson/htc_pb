"""
graph/prepare.py

This file generates Systems.
Each System has one central atom with a randomly-populated
environment surrounding it.
"""

import os
import pickle
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from worker import System

rng = np.random.default_rng(seed=1997)

max_atom_count = 10
n_systems_per_atom_count = {}

for atom_count in range(2, max_atom_count + 1):
    # f(x) = 3861.13*exp(-0.2173x)
    # Chosen so that f(2)=2500 and f(20)=50
    # n_systems_per_atom_count[atom_count] = int(3861.13*np.exp(-0.2173*atom_count))
    n_systems_per_atom_count[atom_count] = int(889.14*np.exp(-0.2878*atom_count))

total_count = sum(n_systems_per_atom_count.values())


def generate_random_values(ranges, size, rng):
    return np.array([rng.uniform(r[0], r[1], size=size) for r in ranges])


def check_overlap(xs, ys, zs, rs):
    indices = np.arange(len(xs))
    for (i, j) in combinations(indices, 2):
        v1 = np.array([xs[i], ys[i], zs[i]])
        v2 = np.array([xs[j], ys[j], zs[j]])
        if np.linalg.norm(v1 - v2) < (rs[i] + rs[j]):
            return True
    return False


def generate_systems(atom_count, n_systems, seed=None):
    """
    Generate a set of systems with a given atom count.
    """
    rng = np.random.default_rng(seed)  # Use a different RNG per task

    inputs = []
    radii_ranges = [(0.8, 2.0)] * atom_count
    charge_ranges = [(-2, 2)] + [(0, 0)] * (atom_count - 1)
    distance_ranges = [(0, 0)] + [(8, 30)] + [(max(r[1] for r in radii_ranges), 8)] * (
        atom_count - 2
    )
    theta_ranges = [(0, 0)] + [(0, np.pi)] * (atom_count - 1)
    phi_ranges = [(0, 2 * np.pi)] * atom_count

    attempts, successful_attempts = 0, 0

    while successful_attempts < n_systems:
        attempts += 1

        rs = generate_random_values(radii_ranges, 1, rng)
        charges = generate_random_values(charge_ranges, 1, rng)
        ds = generate_random_values(distance_ranges, 1, rng)
        thetas = generate_random_values(theta_ranges, 1, rng)
        phis = generate_random_values(phi_ranges, 1, rng)

        xs = ds * np.sin(thetas) * np.cos(phis)
        ys = ds * np.sin(thetas) * np.sin(phis)
        zs = ds * np.cos(thetas)

        if np.isclose(charges[0], 0.0, atol=5e-2) or check_overlap(xs, ys, zs, rs):
            continue

        system = System()
        system.set_radii(rs)
        system.set_charges(charges)
        system.set_positions(np.column_stack((xs, ys, zs)))
        inputs.append(system)
        successful_attempts += 1

    return inputs, attempts


def save_systems(inputs, base_index):
    """
    Save generated systems to files.
    """
    for i, system in enumerate(inputs):
        index = base_index + i
        os.makedirs(f"results/{index}", exist_ok=True)
        os.makedirs(f"worker_logs/{index}", exist_ok=True)
        with open(f"worker_logs/{index}/system_input.pickle", "wb") as f:
            pickle.dump(system, f)


if __name__ == "__main__":
    # Pre-generate unique seeds for each atom count
    seeds = {atom_count: 1997 + atom_count for atom_count in range(2, max_atom_count + 1)}

    with ProcessPoolExecutor(max_workers=24) as executor:
        futures = []
        base_index = 0
        for atom_count in range(2, max_atom_count + 1):
            print(f"Generating {n_systems_per_atom_count[atom_count]} {atom_count}-atom systems...")
            seed = seeds[atom_count]  # Use pre-generated seed
            futures.append(
                executor.submit(
                    generate_systems,
                    atom_count,
                    n_systems_per_atom_count[atom_count],
                    seed=seed,
                )
            )

        for future, atom_count in zip(as_completed(futures), range(2, max_atom_count + 1)):
            inputs, attempts = future.result()
            save_systems(inputs, base_index)
            base_index += len(inputs)
            print(
                f"Completed {atom_count}-atom systems. "
                f"Total attempts: {attempts}.\n"
            )

    print(f"Total count: {total_count} systems")
