import tvm
import os
import numpy as np
from tvm.script import tir as T
from utils import dump
from collections import namedtuple

dtype = "float32"

dir_name = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(dir_name, "__IRModule__/Code3_wrap_tiling")
cuda_path = os.path.join(dir_name, "__cuda__/")


M = N = K = 16384

dim = namedtuple('dim', ['x', 'y', 'z'])
gridDim = dim(x=128, y=128, z=1)
blockDim = dim(x=16, y=16, z=1)
vthreadDim = dim(x=8, y=2, z=1)

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

# print(type(ir_module))
# print(ir_module.script())


'''
read_buffer_index : 0->A 1->B
'''
block_C = sch.get_block("C")
block_shared_A = sch.cache_read(block_C, 0, "shared")
block_local_A = sch.cache_read(block_C, 0, "local")
block_shared_B = sch.cache_read(block_C, 1, "shared")
block_local_B = sch.cache_read(block_C, 1, "local")
block_local_C = sch.cache_write(block_C, 0, "local")

dump(sch.mod["main"].script(), log_path, "0.origin.py")

(i, j, k) = sch.get_loops(block_C)
# by: block_y ; ty: thread_y
by, ty = sch.split(i, factors=[gridDim.y, None])
bx, tx = sch.split(j, factors=[gridDim.x, None])
sch.reorder(by, bx, ty, tx)
sch.bind(by, "blockIdx.y")
sch.bind(bx, "blockIdx.x")

dump(sch.mod["main"].script(), log_path, "1.reorder.py")

# vty: vthread_y ; ty: thread_y; sy: spaital_y
vty, ty, sy = sch.split(ty, factors=[vthreadDim.y, blockDim.y, None])
vtx, tx, sx = sch.split(tx, factors=[vthreadDim.x, blockDim.x, None])

sch.reorder(vty, vtx, ty, tx, sy, sx)

sch.bind(vty, "vthread.y")
sch.bind(vtx, "vthread.x")
sch.bind(ty, "threadIdx.y")
sch.bind(tx, "threadIdx.x")
dump(sch.mod["main"].script(), log_path, "2.thread_bind.py")


sch.reverse_compute_at(block_local_C, tx, preserve_unit_loops=True)
dump(sch.mod["main"].script(), log_path, "3.cache_write_compute_at.py")

ko, ki = sch.split(k, [None, BK])
dump(sch.mod["main"].script(), log_path, "4.split.py")

sch.reorder(ko, ki, sy, sx)

sch.compute_at(block_local_A, ki)
sch.compute_at(block_local_B, ki)
sch.compute_at(block_shared_A, ko)
sch.compute_at(block_shared_B, ko)

'''
sch.compute_at(block_shared_A, ko)
sch.compute_at(block_shared_B, ko)
sch.compute_at(block_local_A, ki)
sch.compute_at(block_local_B, ki)

can not run because:
 The primitive requires all the consumer(s) of the given block to be present under the target loop. However, there are 1 consumer(s) not satisfying the constraint. List of the consumer(s):tir.Block#0
'''


dump(sch.mod["main"].script(), log_path, "5.cache_read_compute_at.py")

aa_yi, aa_xi = sch.get_loops(block_shared_A)[-2:]  # loops size is 7
aa_yi, aa_ty = sch.split(aa_yi, factors=[None, blockDim.y])
aa_xi, aa_tx = sch.split(aa_xi, factors=[None, blockDim.x])
sch.reorder(aa_ty, aa_tx, aa_yi, aa_xi)
sch.bind(aa_ty, "threadIdx.y")
sch.bind(aa_tx, "threadIdx.x")

loops = sch.get_loops(block_shared_B)
bb_yi, bb_xi = sch.get_loops(block_shared_B)[-2:]
bb_yi, bb_ty = sch.split(bb_yi, factors=[None, blockDim.y])
bb_xi, bb_tx = sch.split(bb_xi, factors=[None, blockDim.x])
sch.reorder(bb_ty, bb_tx, bb_yi, bb_xi)
sch.bind(bb_ty, "threadIdx.y")
sch.bind(bb_tx, "threadIdx.x")

sch.decompose_reduction(block_C, ko)

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

dump(cuda_mod.imported_modules[0].get_source(), cuda_path, "Code3_wrap_tiling.cu")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype(dtype), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype(dtype), ctx)
cuda_c = tvm.nd.array(np.zeros((M, N)).astype(dtype), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 6
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))

np.testing.assert_allclose(
    cuda_c.numpy(), cuda_a.numpy() @ cuda_b.numpy(), rtol=1e-3, atol=1e-5)
print("✅✅✅CUDA result is correct")
