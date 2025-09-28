import tvm
import tvm.script.tir as T
import numpy as np

M = K = N = 4096
dtype = "float32"

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
sch = tvm.tir.Schedule(mod)

# gridDim = {4096}, blockDim = {32, 32}
i, j, k = sch.get_loops("C")
i0, i1 = sch.split(i,[None,32])
j0, j1 = sch.split(j,[None,32])
sch.reorder(i0, j0, i1, j1)
sch.bind(i0, "blockIdx.x")
sch.bind(j0, "blockIdx.y")
sch.bind(i1, "threadIdx.x")
sch.bind(j1, "threadIdx.y")

# sch.mod.show()

rt_mod = tvm.build(sch.mod, target="cuda")

cuda_prog = rt_mod.imported_modules[0].get_source()
with open("__cuda__/Code0_native_gemm.cu", "w") as f:
    f.write(cuda_prog)

dev = tvm.cuda(0)
a_np = np.random.uniform(size=(M, K)).astype(dtype)
b_np = np.random.uniform(size=(K, N)).astype(dtype)
c_np = np.zeros((M, N)).astype(dtype)

a = tvm.nd.array(a_np, device=dev)
b = tvm.nd.array(b_np, device=dev)
c = tvm.nd.array(c_np, device=dev)

evaluator = rt_mod.time_evaluator(rt_mod.entry_name, dev, number=10)
timer = evaluator(a, b, c).mean
np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-5, atol=1e-5)

print(f"Time cost: {timer * 1000} ms")
print(f"Native GEMM({M}x{K}x{N}): {2 * M * K * N / timer / 1e12} TFLOPS")
