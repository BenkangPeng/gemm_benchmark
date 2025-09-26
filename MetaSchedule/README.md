This directory includes autotuning GEMM with TVM MetaSchedule.

* tvm version
v0.20.0

**M = K = N = 4096**    **Nvidia/RTX-4090**

| Trials | matmul_AxB.py | matmul_AxB_transpose.py | Cublas        |
| ------ | ------------- | ----------------------- | ------------- |
| 100    | 27.019 TFLOPS |         38.771 TFLOPS                | 50.903 TFLOPS |

