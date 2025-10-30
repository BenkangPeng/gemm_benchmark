"""
This is a 1D version of the GEMM kernel.
The 1D version of the GEMM kernel is a single threadblock kernel.
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
log_path = os.path.join(dir_name, "__IRModule__/Reverse_Engineer/")
cuda_path = os.path.join(dir_name, "__cuda__/Reverse_Engineer/")

M = 4096
K = 4096
N = 4096
dtype = "float32"

# Configuration matching Roller
BM, BN, BK = 128, 128, 8  # Block tile sizes
RM, RN, RK = 16, 8, 1     # Register/vthread tile sizes

# Grid and block dimensions
gridDim_y = M // BM  # 32
gridDim_x = N // BN  # 32
blockDim_y = BM // RM  # 8
blockDim_x = BN // RN  # 16
NUM_BLOCK = gridDim_y * gridDim_x
NUM_THREAD = blockDim_y * blockDim_x


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

# Get the main computation block
block_C = sch.get_block("C")

# Cache reads and writes
block_shared_A = sch.cache_read(block_C, 0, "shared")
block_shared_B = sch.cache_read(block_C, 1, "shared")
block_local_C = sch.cache_write(block_C, 0, "local")

# Get original loops
i, j, k = sch.get_loops(block_C)

# Split spatial dimensions to match Roller's hierarchy:
# i: 4096 = 32 (gridDim.y) * 16 (vthread.x) * 8 (threadDim.y part)
# j: 4096 = 32 (gridDim.x) * 8 (vthread.y) * 16 (threadDim.x part)
i_0, i_1, i_2 = sch.split(i, factors=[gridDim_y, RM, blockDim_y])
j_0, j_1, j_2 = sch.split(j, factors=[gridDim_x, RN, blockDim_x])

# Reorder: block dimensions first, then vthread dimensions, then thread dimensions
# This matches Roller's order
sch.reorder(i_0, j_0, i_1, j_1, i_2, j_2, k)

# Fuse block dimensions
blockIdx_x = sch.fuse(i_0, j_0)

# Fuse thread dimensions
threadIdx_x = sch.fuse(i_2, j_2)

# Bind to GPU hierarchy
sch.bind(blockIdx_x, "blockIdx.x")
sch.bind(i_1, "vthread.x")
sch.bind(j_1, "vthread.y")
sch.bind(threadIdx_x, "threadIdx.x")

# Split reduction dimension: 4096 = 512 * 8
k_0, k_1 = sch.split(k, factors=[None, BK])

# Position shared memory loads and local memory
sch.compute_at(block_shared_A, k_0)
sch.compute_at(block_shared_B, k_0)
sch.reverse_compute_at(block_local_C, threadIdx_x)

# Decompose reduction to create init and update blocks
sch.decompose_reduction(block_C, threadIdx_x)

# Schedule A_shared loading: 128 x 8 = 1024 elements
# Use fused cooperative loading pattern
ax0, ax1 = sch.get_loops(block_shared_A)[-2:]
ax_fused = sch.fuse(ax0, ax1)
# Split into: 2 (unroll) * 128 (threads) * 4 (vectorize) = 1024
ax_outer, ax_mid, ax_inner = sch.split(ax_fused, factors=[None, NUM_THREAD, 4])
sch.unroll(ax_outer)
sch.bind(ax_mid, "threadIdx.x")
sch.vectorize(ax_inner)

# Schedule B_shared loading: 8 x 128 = 1024 elements
# Use fused cooperative loading pattern
ax0, ax1 = sch.get_loops(block_shared_B)[-2:]
ax_fused = sch.fuse(ax0, ax1)
# Split into: 2 (unroll) * 128 (threads) * 4 (vectorize) = 1024
ax_outer, ax_mid, ax_inner = sch.split(ax_fused, factors=[None, NUM_THREAD, 4])
sch.unroll(ax_outer)
sch.bind(ax_mid, "threadIdx.x")
sch.vectorize(ax_inner)

# Optional: Further split k_1 for better register allocation
# This creates the T.grid(8, 1) pattern in Roller
# You can try: k_1_outer, k_1_inner = sch.split(k_1, factors=[8, 1])
# But this might not be necessary depending on TVM version

# Uncomment to see the IR before building
# sch.mod.show()

dump(sch.mod["main"].script(), log_path, "3.vectorize.py")

# Build and test
mod = sch.mod
ctx = tvm.cuda(0)
cuda_mod = tvm.build(mod, target="cuda")

dump(cuda_mod.imported_modules[0].get_source(),
     cuda_path, "Reverse_Engineer.cu")

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
