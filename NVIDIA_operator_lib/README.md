This directory includes some kernels implemented by NVIDIA Operators Library like Cublas, Cutlass and so on.

* `cublas_sgemm.cu`
```shell
$ nvcc cublas_sgemm.cu -lcublas -o bin/cublas_sgemm
$ # test the TFLOPS of special size(M,K,N)
$ ./bin/cublas_sgemm 4096 4096 4096
```