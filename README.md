# pyXBM: Python Extended Bell Measurement

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

pyXBM efficiently evaluates $<\psi_0|A|\psi_1>$ for arbitrary $A\in\mathbb{C}^{2^n\times2^n}$ and $|\psi_0>,|\psi_1>\in\mathbb{C}^{2^n}$ by reducing the number of unique quantum circuits required.

Paper: https://arxiv.org/abs/2110.09735

## Requirements

|Software|Version|
|:---:|:---:|
|Python|3.6.9|
|Qiskit|0.25.0|
|SciPy|1.5.4|

To run Jupyter Notebooks,

|Software|Version|
|:---:|:---:|
|SymPy|1.7.1|
|OpenFermion|1.1.0|
|PyYAML|5.4.1|
|Matplotlib|3.3.4|
|seaborn|0.11.1|
|tqdm|4.57.0|

Download `term_grouping.py` from https://www.github.com/teaguetomesh/vqe-term-grouping/  
Rewrite line 257  
`-   return max_cliques`  
`+   return max_cliques, term_reqs`

## Usage

see [sample.ipynb](https://github2.cae.tytlabs.co.jp/e1689/generalized-bell-measurements/blob/master/sample.ipynb)

```python
from xbm import get_all_meas_circs_bits_coefs
from utils import eval_expectation

psi_0 = ... # qiskit.QuantumCircuit
psi_1 = ... # qiskit.QuantumCircuit
psi_2 = ... # qiskit.QuantumCircuit
mat_A = ... # numpy.ndarray
qins  = ... # qiskit.aqua.QuantumInstance
p_vec = ... # qiskit.circuit.parametervector.ParameterVector, parameter vectors for psi_0, psi_1 and psi_2

meas_1 = get_all_meas_circs_bits_coefs(psi_0, psi_1, mat_A) # <psi_0|A|psi_1>
meas_2 = get_all_meas_circs_bits_coefs(psi_2, None, mat_A @ np.conjugate(mat_A).T) # <psi_2|A^2|psi_2>

params = ... # numpy.ndarray, initial parameters for psi_0, psi_1 and psi_2
for loop in range(nb_loops):
    exp = eval_expectation(meas_1+meas_2, qins, params_dict={p_vec: params})
    params = ... # parameter update
```

## Citing pyXBM

If you find it useful to use this module in your research, please cite the following paper.

```
Ruho Kondo, Yuki Sato, Satoshi Koide, Seiji Kajita and Hideki Takamatsu, Computationally Efficient Quantum Expectation with Extended Bell Measurements, arXiv preprint arXiv:2110.09735, (2021).
```

In Bibtex format:
 
```bibtex
@article{kondo2021xbm,
  title   = {Computationally Efficient Quantum Expectation with Extended Bell Measurements},
  author  = {Kondo, Ruho and Sato, Yuki and Koide, Satoshi and Kajita, Seiji and Takamatsu, Hideki},
  journal = {arXiv preprint arXiv:2110.09735},
  volume  = {},
  pages   = {},
  year    = {2021},
}
```

## License

This project is licensed under the Apache License Version 2.0 - see the LICENSE.txt file for details
