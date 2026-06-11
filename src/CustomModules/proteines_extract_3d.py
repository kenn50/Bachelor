# Code given by Thomas Hammelryck


#!/usr/bin/env python3
"""
pdb_backbone_io.py

Read and write backbone coordinates from multi-model PDB files.

Backbone atom order is always:

    N, CA, C, O

The coordinate array shape is:

    (N, L, 4, 3)

where:

    N = number of models
    L = number of residues per model
    4 = backbone atoms N, CA, C, O
    3 = x, y, z

All written PDB residues are labeled ALA.
"""

from pathlib import Path
import sys

import numpy as np
from Bio.PDB import PDBParser


BACKBONE_ATOMS = ("N", "CA", "C", "O")


def read_backbone_coords(pdb_path):
    """
    Read N, CA, C, O coordinates from a PDB file.

    Parameters
    ----------
    pdb_path:
        Input PDB file.

    Returns
    -------
    np.ndarray
        Array with shape (N, L, 4, 3).

    Notes
    -----
    Residues missing any of N, CA, C, O are skipped.

    All models must have the same number of complete backbone residues.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", str(pdb_path))

    models = []

    for model in structure:
        model_coords = []

        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue

                if not all(atom_name in residue for atom_name in BACKBONE_ATOMS):
                    continue

                residue_coords = [
                    residue[atom_name].get_coord()
                    for atom_name in BACKBONE_ATOMS
                ]

                model_coords.append(residue_coords)

        if len(model_coords) > 0:
            models.append(model_coords)

    if len(models) == 0:
        return np.empty((0, 0, 4, 3), dtype=float)

    lengths = [len(model_coords) for model_coords in models]

    if len(set(lengths)) != 1:
        raise ValueError(
            f"Models have different numbers of complete backbone residues: {lengths}"
        )

    return np.asarray(models, dtype=float)


def save_backbone_npy(coords, npy_path):
    """
    Save backbone coordinates to .npy format.

    Parameters
    ----------
    coords:
        NumPy array with shape (N, L, 4, 3).

    npy_path:
        Output .npy filename.
    """
    coords = np.asarray(coords, dtype=float)
    _check_backbone_shape(coords)

    np.save(npy_path, coords)


def load_backbone_npy(npy_path):
    """
    Load backbone coordinates from .npy format.

    Parameters
    ----------
    npy_path:
        Input .npy filename.

    Returns
    -------
    np.ndarray
        Array with shape (N, L, 4, 3).
    """
    coords = np.load(npy_path)
    _check_backbone_shape(coords)
    return coords


def write_backbone_coords(coords, pdb_path):
    """
    Write backbone coordinates to a multi-model PDB file.

    Parameters
    ----------
    coords:
        NumPy array with shape (N, L, 4, 3).
        Atom order must be N, CA, C, O.

    pdb_path:
        Output PDB filename.

    Notes
    -----
    The output contains N MODEL blocks.
    Each model contains one chain, chain A, with L ALA residues.
    """
    coords = np.asarray(coords, dtype=float)
    _check_backbone_shape(coords)

    n_models, length, _, _ = coords.shape
    pdb_path = Path(pdb_path)

    with open(pdb_path, "w") as f:
        for model_index in range(n_models):
            f.write(f"MODEL     {model_index + 1:4d}\n")

            atom_serial = 1
            chain_id = "A"

            for residue_index in range(length):
                pdb_residue_number = residue_index + 1

                if pdb_residue_number > 9999:
                    raise ValueError("PDB residue numbers cannot exceed 9999")

                for atom_name, xyz in zip(
                    BACKBONE_ATOMS,
                    coords[model_index, residue_index],
                ):
                    x, y, z = xyz
                    element = "C" if atom_name == "CA" else atom_name

                    if atom_name == "CA":
                        atom_field = " CA "
                    else:
                        atom_field = f" {atom_name:<3}"

                    line = (
                        f"ATOM  "
                        f"{atom_serial:5d} "
                        f"{atom_field}"
                        f" "
                        f"ALA "
                        f"{chain_id}"
                        f"{pdb_residue_number:4d}"
                        f"    "
                        f"{x:8.3f}"
                        f"{y:8.3f}"
                        f"{z:8.3f}"
                        f"{1.00:6.2f}"
                        f"{0.00:6.2f}"
                        f"          "
                        f"{element:>2}"
                        f"\n"
                    )

                    f.write(line)
                    atom_serial += 1

            f.write("TER\n")
            f.write("ENDMDL\n")

        f.write("END\n")


def _check_backbone_shape(coords):
    """
    Check that coords has shape (N, L, 4, 3).
    """
    if coords.ndim != 4 or coords.shape[2:] != (4, 3):
        raise ValueError(f"coords must have shape (N, L, 4, 3), got {coords.shape}")


def make_test_coords(n_models=3, length=5):
    """
    Make simple test coordinates with shape (N, L, 4, 3).
    """
    coords = np.zeros((n_models, length, 4, 3), dtype=float)

    for model_index in range(n_models):
        model_shift = model_index * 1.5

        for residue_index in range(length):
            x0 = residue_index * 3.8

            coords[model_index, residue_index, 0] = [
                x0 + 0.0,
                model_shift,
                0.0,
            ]  # N

            coords[model_index, residue_index, 1] = [
                x0 + 1.2,
                model_shift + 0.5,
                0.0,
            ]  # CA

            coords[model_index, residue_index, 2] = [
                x0 + 2.4,
                model_shift,
                0.0,
            ]  # C

            coords[model_index, residue_index, 3] = [
                x0 + 3.0,
                model_shift - 0.6,
                0.0,
            ]  # O

    return coords


if __name__ == "__main__":
    # Test 1: synthetic coordinates -> PDB -> coordinates.
    coords = make_test_coords(n_models=3, length=5)

    test_pdb = Path("test_backbone_models.pdb")
    write_backbone_coords(coords, test_pdb)

    coords_read = read_backbone_coords(test_pdb)

    print("Synthetic PDB round-trip test")
    print("  written PDB:    ", test_pdb)
    print("  original shape: ", coords.shape)
    print("  read shape:     ", coords_read.shape)

    if not np.allclose(coords, coords_read, atol=1e-3):
        raise AssertionError("Synthetic PDB round-trip test failed")

    print("  passed")

    # Test 2: synthetic coordinates -> NPY -> coordinates.
    test_npy = Path("test_backbone_models.npy")
    save_backbone_npy(coords, test_npy)

    coords_npy = load_backbone_npy(test_npy)

    print()
    print("Synthetic NPY round-trip test")
    print("  written NPY:    ", test_npy)
    print("  original shape: ", coords.shape)
    print("  read shape:     ", coords_npy.shape)

    if not np.array_equal(coords, coords_npy):
        raise AssertionError("Synthetic NPY round-trip test failed")

    print("  passed")

    # Test 3: read a real PDB file, write .npy, and write extracted backbone PDB.
    #
    # Usage:
    #     python pdb_backbone_io.py input.pdb
    #
    # Outputs:
    #     input_backbone.npy
    #     input_backbone_ala_models.pdb
    if len(sys.argv) > 1:
        input_pdb = Path(sys.argv[1])

        output_npy = input_pdb.with_name(input_pdb.stem + "_backbone.npy")
        output_pdb = input_pdb.with_name(input_pdb.stem + "_backbone_ala_models.pdb")

        real_coords = read_backbone_coords(input_pdb)

        save_backbone_npy(real_coords, output_npy)
        write_backbone_coords(real_coords, output_pdb)

        print()
        print("Real PDB extraction test")
        print("  input PDB:   ", input_pdb)
        print("  array shape: ", real_coords.shape)
        print("  output NPY:  ", output_npy)
        print("  output PDB:  ", output_pdb)
        print("  check output PDB in PyMOL")
