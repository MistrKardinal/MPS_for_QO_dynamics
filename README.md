# MPS for Quantum-Optical Dynamics

**Numerical simulations of quantum-optical processes using matrix product states (MPS).**

This repository is a research-oriented collection of Python code and Jupyter notebooks for studying the quantum dynamics of optical modes. Its main motivation is to explore how **matrix product states** can represent and evolve multimode photonic states without explicitly storing every amplitude of the full Fock-space wavefunction.

A central example is **spontaneous parametric down-conversion (SPDC)**. The notebooks are intended to make the physical model, numerical procedure, and calculated observables accessible to readers who want to reproduce a simulation and experiment with its parameters.

> **Scope:** This is demonstration / research code rather than a packaged, production-ready quantum simulator. The exact Hamiltonian, mode ordering, propagation algorithm, and parameter conventions should be taken from the corresponding notebook.

## Why matrix product states?

For optical modes with local Fock cutoffs $d_1,\ldots,d_L$, a general pure state can be written as

$$
|\psi\rangle = \sum_{n_1=0}^{d_1-1}\cdots\sum_{n_L=0}^{d_L-1}
C_{n_1\ldots n_L}\,|n_1,\ldots,n_L\rangle.
$$

The coefficient tensor contains $\prod_{j=1}^L d_j$ complex amplitudes. In an MPS representation it is factorized into a chain of smaller tensors:

$$
C_{n_1\ldots n_L}
= A^{[1]n_1} A^{[2]n_2}\cdots A^{[L]n_L},
$$

with adjacent tensors connected by *bond indices*. This can reduce memory requirements substantially when the state admits a good approximation with moderate bond dimensions. The required bond dimensions depend on entanglement and on the chosen ordering and encoding of the optical degrees of freedom.

## What you can explore

- Construct optical quantum states in a truncated Fock basis and represent them as an MPS.
- Follow Hamiltonian-driven quantum evolution in an MPS representation.
- Study SPDC-related dynamics and extract physical observables from the evolving state.
- Experiment with the local Fock-space cutoff, numerical step size, MPS bond dimension, and other model parameters provided in the notebooks.
- Investigate convergence and the trade-off between numerical accuracy and computational resources.

The notebooks provide the entry point; the accompanying Python modules contain reusable numerical routines.

## Get started

### 1. Clone the repository

```bash
git clone https://github.com/MistrKardinal/MPS_for_QO_dynamics.git
cd MPS_for_QO_dynamics
```

### 2. Create a Python environment

Python 3 with `venv` is recommended. On **Windows (PowerShell)**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On **Linux/macOS**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file, install from it:

```bash
python -m pip install -r requirements.txt
```

Otherwise, the following packages provide a starting environment for the Python / PyTorch / Jupyter workflow:

```bash
python -m pip install numpy scipy torch matplotlib jupyterlab
```

Additional packages may be needed depending on the imports in the selected notebook. For a fully reproducible setup, use the exact dependency versions specified by the project when available.

### 4. Open a demonstration notebook

```bash
jupyter lab
```

In JupyterLab, navigate to the SPDC demonstration notebook and run its cells in order. Check the introductory cells for model parameters, Fock cutoffs, tensor dimensions, propagation settings, and output definitions before running a large simulation.

**Tip:** Start with small local Hilbert-space dimensions and modest MPS bond dimensions. Increase them gradually while checking numerical convergence.

## Numerical considerations

An MPS representation does **not** automatically guarantee a compact description. Simulation cost depends on the chosen local basis, entanglement growth, bond dimensions, and implementation of time evolution. In particular:

1. **Fock-space truncation:** Increasing photon-number cutoffs is necessary when probability weight approaches a basis boundary.
2. **Bond-dimension convergence:** Increasing the allowed MPS bond dimension helps assess tensor-network truncation effects.
3. **Time-discretization error:** Repeat calculations with smaller time steps to test temporal convergence.
4. **Conservation laws and normalization:** Where applicable, monitor norm and conserved quantities as numerical diagnostics.
5. **Mode ordering:** The placement of degrees of freedom along the MPS chain can have a strong effect on required bond dimensions.

For reproducible comparisons, report the physical parameters alongside the Fock cutoffs, MPS truncation settings, time step, and hardware used.

## Repository status and contributions

The code is shared as an evolving demonstration of tensor-network methods for quantum optics. Suggestions, bug reports, and reproducible examples are welcome through [GitHub Issues](https://github.com/MistrKardinal/MPS_for_QO_dynamics/issues).

If you use or modify the code, please also check the repository's `LICENSE` file (if present) for the applicable terms.
