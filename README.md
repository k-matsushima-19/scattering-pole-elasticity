# scattering-pole-elasticity
Python code for computing scattering poles of layered cylindrical systems

See [my preprint](https://arxiv.org/abs/2302.10231) or [published article](https://link.aps.org/doi/10.1103/PhysRevB.107.144104) for details.

## Required packages
- [Numpy](https://pypi.org/project/numpy/)
- [Scipy](https://pypi.org/project/scipy/)
- [tqdm](https://pypi.org/project/tqdm/)

Tested on Python 3.8.10 with Numpy 1.23.2, Scipy 1.8.0, and tqdm 4.64.0

## How to use
Just execute the main script.
```bash
python3 main.py
```
You can change parameters in main.py

## References
- Kei Matsushima, Yuki Noguchi, Takayuki Yamada: [Exceptional points in cylindrical elastic media with radiation loss](https://link.aps.org/doi/10.1103/PhysRevB.107.144104)
- Junko Asakura, Tetsuya Sakurai, Hiroto Tadano, Tsutomu Ikegami, Kinji Kimura
: [A numerical method for nonlinear eigenvalue problems using contour integrals](https://www.jstage.jst.go.jp/article/jsiaml/1/0/1_0_52/_article/-char/en)
