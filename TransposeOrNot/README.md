There are no performance gap after tiling optimization(`matmul_A_B.py` `matmul_AT_B.py` `matmul_ATT_B.py` `matmul_A_B_transpose.py`).

<del> But the performance gap is clearly observable in `performance_diff.py` without tiling optimization.</del>

Found the culprit! 🚨 The slowdown in `matmul_A_B` is due to the poor thread mapping. Refer to https://github.com/BenkangPeng/gemm_benchmark/issues/1

And I don't know why......
