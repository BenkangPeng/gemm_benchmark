import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
import tvm.meta_schedule as ms
from tvm.meta_schedule.database.json_database import JSONDatabase
from collections import namedtuple

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--M", type=int, default=4096)
parser.add_argument("--K", type=int, default=4096)
parser.add_argument("--N", type=int, default=4096)
args = parser.parse_args()
M = args.M
K = args.K
N = args.N
dtype = "float32"

assert M == K

# compute C = A @ B


@tvm.script.ir_module
class matmul_A_B:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="float32")
        B = T.match_buffer(b, [K, N], dtype="float32")
        C = T.match_buffer(c, [M, N], dtype="float32")
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vk, vi] * B[vk, vj]


sch = tvm.tir.Schedule(matmul_A_B)
dim3 = namedtuple('dim3', ['x', 'y', 'z'])
gridDim = dim3(x=128, y=128, z=1)
blockDim = dim3(x=16, y=16, z=1)

block_C = sch.get_block("C")
i, j, k = sch.get_loops(block_C)
i0, i1, i2 = sch.split(i, factors=[gridDim.y, blockDim.y, None])
j0, j1, j2 = sch.split(j, factors=[gridDim.x, blockDim.x, None])
sch.reorder(i0, j0, i1, j1, i2, j2)
sch.bind(i0, "blockIdx.y")
sch.bind(j0, "blockIdx.x")
sch.bind(i1, "threadIdx.y")
sch.bind(j1, "threadIdx.x")

BK = 16
k0, k1 = sch.split(k, factors=[None, BK])
sch.reorder(k0, k1, i2, j2)

block_A_shared = sch.cache_read(block_C, 0, "shared")
block_A_local = sch.cache_read(block_C, 0, "local")
block_B_shared = sch.cache_read(block_C, 1, "shared")
block_B_local = sch.cache_read(block_C, 1, "local")
block_C_local = sch.cache_write(block_C, 0, "local")

sch.compute_at(block_A_local, k1)
sch.compute_at(block_B_local, k1)
sch.compute_at(block_A_shared, k0)
sch.compute_at(block_B_shared, k0)

sch.reverse_compute_at(block_C_local, j2)

i, j = sch.get_loops(block_A_shared)[-2:]
i0, i1 = sch.split(i, factors=[blockDim.y, None])
j0, j1 = sch.split(j, factors=[blockDim.x, None])
sch.reorder(i0, j0, i1, j1)
sch.bind(i0, "threadIdx.y")
sch.bind(j0, "threadIdx.x")

i, j = sch.get_loops(block_B_shared)[-2:]
i0, i1 = sch.split(i, factors=[blockDim.y, None])
j0, j1 = sch.split(j, factors=[blockDim.x, None])
sch.reorder(i0, j0, i1, j1)
sch.bind(i0, "threadIdx.y")
sch.bind(j0, "threadIdx.x")

sch.decompose_reduction(block_C, k0)


# sch.mod.show()

rt_mod = tvm.build(sch.mod, target="cuda")

dev = tvm.cuda(0)
a_np = np.random.uniform(size=(M, K)).astype(dtype)
b_np = np.random.uniform(size=(K, N)).astype(dtype)
c_np = np.zeros((M, N)).astype(dtype)

a = tvm.nd.array(a_np, device=dev)
b = tvm.nd.array(b_np, device=dev)
c = tvm.nd.array(c_np, device=dev)

evaluator = rt_mod.time_evaluator(rt_mod.entry_name, dev, number=10)
timer = evaluator(a, b, c).mean

print(f"Time cost: {timer * 1000} ms")
print(f"Blocked GEMM({M}x{K}x{N}): {2 * M * K * N / timer / 1e12} TFLOPS")
np.testing.assert_allclose(c.numpy(), a_np.T @ b_np, rtol=1e-3, atol=1e-3)
