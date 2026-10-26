"""Shared experiment harness for the equivariance-guided defense work (see IMPLEMENTATION_PLAN.md)."""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUI4ROB_DIR = os.path.join(ROOT_DIR, "Equi4Rob")
if EQUI4ROB_DIR not in sys.path:
    sys.path.insert(0, EQUI4ROB_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "cityscapes_dataset_subsampled")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "improvements")
