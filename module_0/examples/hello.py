"""
hello.py
========

Trivial Python script used in Module 0 / Lecture 3 to demonstrate end-to-end
SLURM submission on ARCC.

Prints host, Python version, working directory, and SLURM environment variables
so students can verify they actually landed on a compute node (not a login node).
"""

import os
import socket
import sys


def main() -> None:
    print("Hello from ARCC.")
    print(f"Hostname:           {socket.gethostname()}")
    print(f"Python version:     {sys.version.split()[0]}")
    print(f"Working directory:  {os.getcwd()}")
    print(f"SLURM_JOB_ID:       {os.environ.get('SLURM_JOB_ID', 'none')}")
    print(f"SLURM_NODELIST:     {os.environ.get('SLURM_NODELIST', 'none')}")
    print(f"SLURM_CPUS_ON_NODE: {os.environ.get('SLURM_CPUS_ON_NODE', 'none')}")


if __name__ == "__main__":
    main()
