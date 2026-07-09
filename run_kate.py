#!/usr/bin/env python
"""Top-level entry point for KATE.

A thin wrapper around the installed console script so the tool can be run from a
bare checkout without the entry point on PATH:

    python run_kate.py compress topology.pdb traj.dcd -o run.kate
    python run_kate.py analyze run.kate --mfpt 2 --bootstrap
    python run_kate.py bound run.kate ref.kate

It forwards straight to `kate.cli:main`, so it is equivalent to the `kate` command.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kate.cli import main

if __name__ == "__main__":
    sys.exit(main())
