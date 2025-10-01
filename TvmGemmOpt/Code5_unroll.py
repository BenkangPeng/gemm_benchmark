import tvm
import os
import numpy as np
from tvm.script import tir as T
from utils import dump
from collections import namedtuple

dtype = "float32"

dir_name = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(dir_name, "__IRModule__/Code5_unroll")
cuda_path = os.path.join(dir_name, "__cuda__/")

M = N = K = 16384

dim3 = namedtuple('dim3', ['x', 'y', 'z'])
gridDim = dim3(x=128, y=128, z=1)
blockDim = dim3(x=16, y=16, z=1)
vthreadDim = dim3(x=2, y=2, z=1)

BK = 16


@tvm.script.ir_module
class GEMM:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K])
        B = T.match_buffer(b, [K, N])
        C = T.match_buffer(c, [M, N])

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
block_local_A = sch.cache_read(block_C, 0, "local")
block_shared_B = sch.cache_read(block_C, 1, "shared")
block_local_B = sch.cache_read(block_C, 1, "local")
block_local_C = sch.cache_write(block_C, 0, "local")

dump(sch.mod["main"].script(), log_path, "0.origin.py")

i, j, k = sch.get_loops(block_C)
# by: block_y ; vty: vthread_y ; ty: thread_y ; sy: spatial_y
by, vty, ty, sy = sch.split(
    i, factors=[gridDim.y, vthreadDim.y, blockDim.y, None])
# bx: block_x ; vtx: vthread_x ; tx: thread_x ; sx: spatial_x
bx, vtx, tx, sx = sch.split(
    j, factors=[gridDim.x, vthreadDim.x, blockDim.x, None])
sch.reorder(by, bx, vty, vtx, ty, tx, sy, sx)
sch.bind(by, "blockIdx.y")
sch.bind(bx, "blockIdx.x")
sch.bind(vty, "vthread.y")
sch.bind(vtx, "vthread.x")
sch.bind(ty, "threadIdx.y")
sch.bind(tx, "threadIdx.x")
dump(sch.mod["main"].script(), log_path, "1.binding.py")

sch.reverse_compute_at(block_local_C, tx, preserve_unit_loops=True)
k0, k1 = sch.split(k, factors=[None, BK])
sch.reorder(k0, k1, sy, sx)
sch.decompose_reduction(block_C, tx)
dump(sch.mod["main"].script(), log_path, "2.split.py")


sch.compute_at(block_local_A, k1)
sch.compute_at(block_local_B, k1)
sch.compute_at(block_shared_A, k0)
sch.compute_at(block_shared_B, k0)

dump(sch.mod["main"].script(), log_path, "3.compute_at.py")

# Use all the threads in the same threadblock together to load A tile(BM x BK) to shared memory
ax0, ax1 = sch.get_loops(block_shared_A)[-2:]
ax0, ax1_ty = sch.split(ax0, factors=[None, blockDim.y])
ax1, ax1_tx = sch.split(ax1, factors=[None, blockDim.x])
sch.reorder(ax1_ty, ax1_tx, ax0, ax1)
sch.bind(ax1_ty, "threadIdx.y")
sch.bind(ax1_tx, "threadIdx.x")
# ax_s : axis spatial
ax_s = sch.fuse(ax0, ax1)
ax_s, ax_f4 = sch.split(ax_s, factors=[None, 4])
sch.vectorize(ax_f4)  # use cuda float4 instruction to load 4 floats at a time

ax0, ax1 = sch.get_loops(block_shared_B)[-2:]
ax0, ax1_ty = sch.split(ax0, factors=[None, blockDim.y])
ax1, ax1_tx = sch.split(ax1, factors=[None, blockDim.x])
sch.reorder(ax1_ty, ax1_tx, ax0, ax1)
sch.bind(ax1_ty, "threadIdx.y")
sch.bind(ax1_tx, "threadIdx.x")
# ax_s : axis spatial
ax_s = sch.fuse(ax0, ax1)
ax_s, ax_f4 = sch.split(ax_s, factors=[None, 4])
sch.vectorize(ax_f4)  # use cuda float4 instruction to load 4 floats at a time

sch.vectorize(sch.get_loops(block_local_A)[-1])
sch.vectorize(sch.get_loops(block_local_B)[-1])
dump(sch.mod["main"].script(), log_path, "4.vectorize.py")

# unroll some loops of big extent
# k0_u: k0_unroll
k0_u, k0 = sch.split(k0, factors=[16, None])
sch.unroll(k0_u)
# NOTE Avoid unrolling inner loops — it will bloat code size and extend compile time without meaningful gains.
# sch.unroll(k1) # too long compile time
dump(sch.mod["main"].script(), log_path, "5.unroll.py")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

dump(cuda_mod.imported_modules[0].get_source(), cuda_path, "Code5_unroll.cu")

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
