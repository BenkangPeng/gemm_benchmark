import tvm
import numpy as np
import tvm.script.tir as T
from collections import namedtuple

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
dim3 = namedtuple('dim3', ['x', 'y', 'z'])
blockDim = dim3(x=16, y=16, z=1)
gridDim = dim3(x=M//16, y=N//16, z=1)


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
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# compute C = A @ B by first computing AT = A^T, then C = AT @ B
@tvm.script.ir_module
class matmul_A_B_transpose:
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

        # Compute C = (AT)T @ B = A @ B
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + AT[vk, vi] * B[vk, vj]


def sch_A_B(mod: tvm.IRModule) -> tvm.tir.Schedule:
    sch = tvm.tir.Schedule(mod)
    block_C = sch.get_block("C")
    i, j, k = sch.get_loops(block_C)
    bi, ti, si = sch.split(i, [gridDim.y, blockDim.y, None])
    bj, tj, sj = sch.split(j, [gridDim.x, blockDim.x, None])
    sch.reorder(bi, bj, ti, tj, si, sj)
    sch.bind(bi, "blockIdx.y")
    sch.bind(bj, "blockIdx.x")
    sch.bind(ti, "threadIdx.y")
    sch.bind(tj, "threadIdx.x")

    # poor performance!!!
    # refer to https://github.com/BenkangPeng/gemm_benchmark/issues/1
    # sch.bind(bi, "blockIdx.x")
    # sch.bind(bj, "blockIdx.y")
    # sch.bind(ti, "threadIdx.x")
    # sch.bind(tj, "threadIdx.y")
    return sch


def sch_A_B_trans(mod: tvm.IRModule) -> tvm.tir.Schedule:
    sch = tvm.tir.Schedule(mod)
    block_AT = sch.get_block("AT")
    i, j = sch.get_loops(block_AT)
    bi, ti, si = sch.split(i, [gridDim.y, blockDim.y, None])
    bj, tj, sj = sch.split(j, [gridDim.x, blockDim.x, None])
    sch.reorder(bi, bj, ti, tj, si, sj)
    sch.bind(bi, "blockIdx.y")
    sch.bind(bj, "blockIdx.x")
    sch.bind(ti, "threadIdx.y")
    sch.bind(tj, "threadIdx.x")
    block_C = sch.get_block("C")
    i, j, k = sch.get_loops(block_C)
    bi, ti, si = sch.split(i, [gridDim.y, blockDim.y, None])
    bj, tj, sj = sch.split(j, [gridDim.x, blockDim.x, None])
    sch.reorder(bi, bj, ti, tj, si, sj)
    sch.bind(bi, "blockIdx.y")
    sch.bind(bj, "blockIdx.x")
    sch.bind(ti, "threadIdx.y")
    sch.bind(tj, "threadIdx.x")
    return sch


if __name__ == "__main__":
    sch0 = sch_A_B(matmul_A_B)
    sch1 = sch_A_B_trans(matmul_A_B_transpose)

    rt_mod0 = tvm.build(sch0.mod, target="cuda")
    rt_mod1 = tvm.build(sch1.mod, target="cuda")

    dev = tvm.cuda(0)
    a_np = np.random.uniform(size=(M, K)).astype(dtype)
    b_np = np.random.uniform(size=(K, N)).astype(dtype)
    c_np = np.zeros((M, N)).astype(dtype)
    a = tvm.nd.array(a_np, device=dev)
    b = tvm.nd.array(b_np, device=dev)
    c = tvm.nd.array(c_np, device=dev)
    at = tvm.nd.array(a_np.T, device=dev)

    evaluator0 = rt_mod0.time_evaluator(rt_mod0.entry_name, dev, number=10)
    evaluator1 = rt_mod1.time_evaluator(rt_mod1.entry_name, dev, number=10)

    timer0 = evaluator0(a, b, c).mean
    timer1 = evaluator1(a, b, c, at).mean

    print(f"Time cost: {timer0 * 1000} ms")
    print(f"Time cost: {timer1 * 1000} ms")
