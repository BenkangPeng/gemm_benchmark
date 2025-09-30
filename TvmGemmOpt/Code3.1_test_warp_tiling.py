import tvm
import tvm.script.tir as T
from collections import namedtuple
import numpy as np
from utils import dump
import os

dtype = "float32"
M = N = 2048
dim = namedtuple('dim', ['x', 'y', 'z'])
gridDim = dim(x=1, y=1, z=1)
blockDim = dim(x=16, y=16, z=1)

assert M == N
assert M % blockDim.x == 0 and N % blockDim.y == 0
vthreadDim = dim(x=M // blockDim.x, y=N // blockDim.y, z=1)

IRModule_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "__IRModule__/Code3.1_test_warp_tiling")
cuda_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "__cuda__/")


@tvm.script.ir_module
class transpose:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, N])
        B = T.match_buffer(b, [N, M])
        for i, j in T.grid(N, M):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vj, vi]


sch = tvm.tir.Schedule(transpose)
block_B = sch.get_block("B")
i, j = sch.get_loops(block_B)
i0, i1 = sch.split(i, factors=[blockDim.y, None])
j0, j1 = sch.split(j, factors=[blockDim.x, None])
sch.reorder(i0, j0, i1, j1)
sch.bind(i0, "threadIdx.y")
sch.bind(j0, "threadIdx.x")

# sch.mod.show()

rt_mod = tvm.build(sch.mod, target="cuda")
dump(sch.mod["main"].script(), IRModule_path,
     "Code3.1_test_warp_tiling_without_vthread.py")
dump(rt_mod.imported_modules[0].get_source(), cuda_path,
     "Code3.1_test_warp_tiling_without_vthread.cu")

dev = tvm.cuda(0)
a_np = np.random.uniform(size=(M, N)).astype(dtype)
b_np = np.zeros((N, M)).astype(dtype)

a = tvm.nd.array(a_np, device=dev)
b = tvm.nd.array(b_np, device=dev)

evaluator = rt_mod.time_evaluator(rt_mod.entry_name, dev, number=10)
timer = evaluator(a, b).mean
np.testing.assert_allclose(b.numpy(), a_np.T, rtol=1e-3, atol=1e-3)

print(f"Time cost without vthread: {timer * 1000} ms")

##################################################
# Use vthread to tile the warp
sch = tvm.tir.Schedule(transpose)
block_B = sch.get_block("B")
i, j = sch.get_loops(block_B)
i0, i1 = sch.split(i, factors=[vthreadDim.y, blockDim.y])
j0, j1 = sch.split(j, factors=[vthreadDim.x, blockDim.x])
sch.reorder(i0, j0, i1, j1)
sch.bind(i0, "vthread.y")
sch.bind(j0, "vthread.x")
sch.bind(i1, "threadIdx.y")
sch.bind(j1, "threadIdx.x")

rt_mod = tvm.build(sch.mod, target="cuda")
dump(sch.mod["main"].script(), IRModule_path,
     "Code3.1_test_warp_tiling_with_vthread.py")
dump(rt_mod.imported_modules[0].get_source(),
     cuda_path, "Code3.1_test_warp_tiling_with_vthread.cu")

evaluator = rt_mod.time_evaluator(rt_mod.entry_name, dev, number=10)
timer = evaluator(a, b).mean
np.testing.assert_allclose(b.numpy(), a_np.T, rtol=1e-3, atol=1e-3)

print(f"Time cost with vthread: {timer * 1000} ms")
