This folder is the benchmark of paper: [Accelerated Auto-Tuning of GPU Kernels for Tensor Computations](https://dl.acm.org/doi/10.1145/3650200.3656626)

Usage
```shell
# the python env below is built following https://github.com/HPCRL/tvm
# But it seems that `HPCRL/tvm` only implements `dynamic gradient search` and lacks `Ansor-AF`.
$ conda activate ics24
$ python DynamicSearch.py --M 512 --N 1024 --K 4096 --tune
```