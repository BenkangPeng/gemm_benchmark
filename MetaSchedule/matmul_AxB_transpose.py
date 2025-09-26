"""
A non-standard matmul implementation:
    - Firstly transpose A to AT, then do the matmul of (AT)^T and B.

But it outperforms the standard matmul(A @ B) for no reason.(And I don't know why)

* tvm version
v0.20.0
"""

"""
compute C = A @ B by first computing AT = A^T, then C = (AT)^T @ B = A @ B

"""

import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
import tvm.meta_schedule as ms
from tvm.meta_schedule.database import JSONDatabase
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--M", type=int, default=4096)
parser.add_argument("--K", type=int, default=4096)
parser.add_argument("--N", type=int, default=4096)
parser.add_argument("--tune", type=int, default=1,
                    help="1 to tune, 0 to pick the best schedule from the database")
args = parser.parse_args()
M = args.M
K = args.K
N = args.N
tune = args.tune


# compute C = A @ B by first computing AT = A^T, then C = (AT)^T @ B
@tvm.script.ir_module
class matmul_A_B:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle, at: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="float32")
        B = T.match_buffer(b, [K, N], dtype="float32")
        C = T.match_buffer(c, [M, N], dtype="float32")
        AT = T.match_buffer(at, [K, M], dtype="float32")

        # Transpose A to get AT
        for i, j in T.grid(K, M):
            with T.block("AT"):
                vi, vj = T.axis.remap("SS", [i, j])
                AT[vi, vj] = A[vj, vi]

        # Compute C = (AT)^T @ B = A @ B
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + AT[vk, vi] * B[vk, vj]



work_dir = "__tune_tmp__"
target = "nvidia/geforce-rtx-4090"

if tune:
    database = ms.tune_tir(
        mod=matmul_A_B,
        target=target,
        work_dir=work_dir,
        # max times to run the candidate programs in device(e.g. GPU)
        max_trials_global=100,
        # num_trials_per_iter=10
    )
else:
    path_workload = os.path.join(work_dir, "database_workload.json")
    path_tuning_record = os.path.join(work_dir, "database_tuning_record.json")
    database = JSONDatabase(path_workload, path_tuning_record, work_dir=work_dir)


sch = ms.tir_integration.compile_tir(database, matmul_A_B, target)
# sch.mod.show()
# exit()

rt_mod = tvm.build(sch.mod, "cuda")

# cuda_prog = rt_mod.imported_modules[0].get_source()
# with open("matmul_A_B_transpose.cu", "w") as f:
#     f.write(cuda_prog)

a_np = np.random.uniform(size=(M, K)).astype("float32")
b_np = np.random.uniform(size=(K, N)).astype("float32")
a_nd = tvm.nd.array(a_np, tvm.cuda(0))
b_nd = tvm.nd.array(b_np, tvm.cuda(0))
c_nd = tvm.nd.array(np.zeros((M, N), dtype="float32"), tvm.cuda(0))
at_nd = tvm.nd.array(np.zeros((K, M), dtype="float32"), tvm.cuda(0))

rt_mod["main"](a_nd, b_nd, c_nd, at_nd)
np.testing.assert_allclose(c_nd.numpy(), a_np @ b_np, rtol=1e-5, atol=1e-5)

num_flop = 2 * M * K * N
evaluator = rt_mod.time_evaluator(rt_mod.entry_name, tvm.cuda(0), number=10)
timer = evaluator(a_nd, b_nd, c_nd,at_nd).mean
print("time cost on average: %f ms" % (timer * 1e3))
print("GEMM-Blocking: %f GFLOPS" %
      (num_flop / timer / 1e9))
