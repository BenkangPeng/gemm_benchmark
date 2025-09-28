import tvm
import tvm.script.tir as T
import numpy as np
from collections import namedtuple
import os
from utils import dump

M = K = N = 16384
dtype = "float32"
dir_name = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(dir_name, "__IRModule__/Code2_threadtiling")
cuda_path = os.path.join(dir_name, "__cuda__/")

@tvm.script.ir_module
class GEMM:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (M, K), dtype=dtype)
        B = T.match_buffer(b, (K, N), dtype=dtype)
        C = T.match_buffer(c, (M, N), dtype=dtype)
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


mod = GEMM

dim = namedtuple('dim', ['x', 'y', 'z'])
gridDim = dim(x=128, y=128, z=1)
blockDim = dim(x=16, y=16, z=1)

sch = tvm.tir.Schedule(mod)
block_C = sch.get_block("C")

# Read A from global memory to shared memory
# Buffer `C` read 2 buffers, A and B; index: (0->A 1->B)
block_shared_A = sch.cache_read(block_C, 0, "shared")
# Read A from shared memory to local memory
block_local_A = sch.cache_read(block_C, 0, "local")

block_shared_B = sch.cache_read(block_C, 1, "shared")
block_local_B = sch.cache_read(block_C, 1, "local")

# cache write C to local memory(register)
block_local_C = sch.cache_write(block_C, 0, "local")

i, j, k = sch.get_loops(block_C)


i0, i1, i2 = sch.split(i, factors=[gridDim.y, blockDim.y, None])
j0, j1, j2 = sch.split(j, factors=[gridDim.x, blockDim.x, None])
sch.reorder(i0, j0, i1, j1, i2, j2)
sch.reverse_compute_at(block_local_C, j1)

sch.bind(i0, "blockIdx.y")
sch.bind(j0, "blockIdx.x")
sch.bind(i1, "threadIdx.y")
sch.bind(j1, "threadIdx.x")

# the order of thread binding above outperforms the order below:
# sch.bind(bi, "blockIdx.x")
# sch.bind(bj, "blockIdx.y")
# sch.bind(ti, "threadIdx.x")
# sch.bind(tj, "threadIdx.y")

dump(sch.mod["main"].script(), log_path, "0_threadbinding.py")

BK = 16
k0, k1 = sch.split(k, factors=[None, BK])
sch.reorder(k0, k1, i2, j2)
dump(sch.mod["main"].script(), log_path, "1_split_reorder_k.py")

sch.compute_at(block_local_A, k1)
sch.compute_at(block_local_B, k1)
sch.compute_at(block_shared_A, k0)
sch.compute_at(block_shared_B, k0)
dump(sch.mod["main"].script(), log_path, "2_compute_at.py")

# Threads in the same threadblock collaboratively  load A from global memory to shared memory together
# aka Coalesced load
ax0, ax1 = sch.get_loops(block_shared_A)[-2:]
ax0, ax1_ty = sch.split(ax0, factors=[None, blockDim.y])
ax1, ax1_tx = sch.split(ax1, factors=[None, blockDim.x])
sch.reorder(ax1_ty, ax1_tx, ax0, ax1)
sch.bind(ax1_ty, "threadIdx.y")
sch.bind(ax1_tx, "threadIdx.x")
dump(sch.mod["main"].script(), log_path, "3_thread_bind.py")

ax0, ax1 = sch.get_loops(block_shared_B)[-2:]
ax0, ax1_ty = sch.split(ax0, factors=[None, blockDim.y])
ax1, ax1_tx = sch.split(ax1, factors=[None, blockDim.x])
sch.reorder(ax1_ty, ax1_tx, ax0, ax1)
sch.bind(ax1_ty, "threadIdx.y")
sch.bind(ax1_tx, "threadIdx.x")
dump(sch.mod["main"].script(), log_path, "4_thread_bind.py")

sch.decompose_reduction(block_C, k0)
dump(sch.mod["main"].script(), log_path, "5_decompose_reduction.py")

rt_mod = tvm.build(sch.mod, target="cuda")
dump(rt_mod.imported_modules[0].get_source(), cuda_path, "Code2_threadtiling.cu")

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
np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-3, atol=1e-3)
