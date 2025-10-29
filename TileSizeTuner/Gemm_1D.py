"""
This is a 1D version of the GEMM kernel.
The 1D version of the GEMM kernel is a single threadblock kernel.
The threadblock is a 1D threadblock.
The threadblock is a 1D threadblock.
"""
import tvm
import os
from tvm.script import tir as T
from tvm.script import ir as I
from collections import namedtuple
from utils import dump
import numpy as np

dir_name = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(dir_name, "__IRModule__/Gemm_1D/")
cuda_path = os.path.join(dir_name, "__cuda__/Gemm_1D/")

M = 4096
K = 4096
N = 4096
dtype = "float32"

dim3 = namedtuple('dim3', ['x', 'y', 'z'])
rtile0 = [128, 128, 8] # [BM, BN, BK]
rtile1 = [16, 8, 1] # [RM, RN, RK]
gridDim = dim3(x=N//rtile0[1] , y = M//rtile0[0], z=1)
blockDim = dim3(x=rtile0[1]//rtile1[1], y=rtile0[0]//rtile1[0], z=1)
BK = rtile0[2]


@I.ir_module
class GEMM:
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
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


mod = GEMM
sch = tvm.tir.Schedule(mod)

block_C = sch.get_block("C")
block_shared_A = sch.cache_read(block_C, 0, "shared")
block_shared_B = sch.cache_read(block_C, 1, "shared")
block_local_C = sch.cache_write(block_C, 0, "local")

i, j, k = sch.get_loops(block_C)
by, ty, vy = sch.split(i, factors=[gridDim.y, blockDim.y, None])
bx, tx, vx = sch.split(j, factors=[gridDim.x, blockDim.x, None])
sch.reorder(by, bx, vy, vx, ty, tx)
blockIdx_x = sch.fuse(by, bx)
threadIdx_x = sch.fuse(ty, tx)

sch.bind(blockIdx_x, "blockIdx.x")
sch.bind(vy, "vthread.x")
sch.bind(vx, "vthread.y")
sch.bind(threadIdx_x, "threadIdx.x")
k0, k1 = sch.split(k, factors=[None, BK])

sch.compute_at(block_shared_A, k0)
sch.compute_at(block_shared_B, k0)
sch.reverse_compute_at(block_local_C, threadIdx_x)
sch.decompose_reduction(block_C, threadIdx_x)

# threads in the same threadblock collaboratively load A tile(BM x BK) to shared memory
ax0, ax1 = sch.get_loops(block_shared_A)[-2:]
sch.bind(ax0, "threadIdx.x")
ax1_unroll, ax1_vectorize = sch.split(ax1, factors=[None, 4])
sch.unroll(ax1_unroll)
sch.vectorize(ax1_vectorize)

# threads in the same threadblock collaboratively load B tile(BK x BN) to shared memory
ax0, ax1 = sch.get_loops(block_shared_B)[-2:]
sch.reorder(ax1, ax0)
sch.bind(ax1, "threadIdx.x")
ax0_unroll, ax0_vectorize = sch.split(ax0, factors=[None, 4])
sch.unroll(ax0_unroll)
sch.vectorize(ax0_vectorize)

# sch.mod.show()
# exit()

dump(sch.mod["main"].script(), log_path, "3.vectorize.py")
# exit()

mod = sch.mod
# from Roller_IRMod import Module
# mod = Module
ctx = tvm.cuda(0)
cuda_mod = tvm.build(mod, target="cuda")

dump(cuda_mod.imported_modules[0].get_source(),
     cuda_path, "Gemm_1D.cu")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype(dtype), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype(dtype), ctx)
cuda_c = tvm.nd.array(np.zeros((M, N)).astype(dtype), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 10
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
np.testing.assert_allclose(
    cuda_c.numpy(), cuda_a.numpy() @ cuda_b.numpy(), rtol=1e-3, atol=1e-5)
print("✅✅✅CUDA result is correct")
