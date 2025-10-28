import tvm
import tvm.testing
from tvm import auto_scheduler
import numpy as np
import argparse
import tvm.te as te
import os

args = argparse.ArgumentParser()
args.add_argument("--M", type=int, default=512)
args.add_argument("--K", type=int, default=1024)
args.add_argument("--N", type=int, default=4096)
args.add_argument("--tune", action="store_true", help="tune the kernel or not")
args = args.parse_args()
M, K, N = args.M, args.K, args.N
tune = args.tune

target = tvm.target.Target("cuda")
work_dir = os.path.dirname(os.path.abspath(__file__))
tune_dir = os.path.join(work_dir, "__tune__")

IRModule_dir = os.path.join(work_dir, "__IRModule__")
CUDA_dir = os.path.join(work_dir, "__CUDA__")
tune_log_file = os.path.join(tune_dir, "tune.log")


def dump(content: str, path: str, fname: str):
    """
    Dump code to file
    Args:
        content: str, the content to dump
        path: str, the path to dump
        fname: str, the name of the file
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, fname), "w") as f:
        f.write(content)

@auto_scheduler.register_workload
def mamtul(M, K, N):
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]),
        name="C",
        attrs={"layout_free_placeholders": [B]},
    )
    return [A, B, C]


task = auto_scheduler.SearchTask(
    func=mamtul, args=(M, K, N), target=target
)

print(f'computation graph of task:\n {task.compute_dag}')

slide_window_size = 10  # Size of the sliding window used in dynamic gradient search
max_tuning_time = 120  # Maximum tuning time in seconds, 120 is the suggested value
max_trials = 100  # Maximum number of measurement trials to perform in dynamic gradient search, use 1000 to get better performance
n_start = 5  # Number of start points from the initial sampled population
init_size = (
    5  # Number of samples to generate the initial model, 64 is the suggested value
)
predict_score_threshold_ratio = 0.6  # Threshold for the predict score
measure_threshold_ratio = 0.6  # Threshold for the measured throughput

# Tuning options, tested with local runner and builder
tune_option = auto_scheduler.TuningOptions(
    runner=auto_scheduler.LocalRunner(timeout=10),
    builder=auto_scheduler.LocalBuilder(timeout=10),
)

# Initialize tuner
tuner = auto_scheduler.dynamic_gradient_search.DynamicGradientSearchTuner(
    task,
    tune_log_file,
    tune_option,
    n_start,
    init_size,
    slide_window_size,
    max_trials,
    max_tuning_time,
    predict_score_threshold_ratio,
    measure_threshold_ratio,
)

# running tuning task
if tune:
    tuner.dynamic_gradient_search()

# select the best schedule from the tuning log
sch, args = task.apply_best(tune_log_file)
assert sch is not None and "Do not find the any schedule in the tuning log. Please tuning the kernel first."
# print(sch.stage_map)
# print(args)
mod = tvm.lower(sch, args, simple_mode=True)
func = tvm.build(mod, args, target)
best_cuda_program = func.imported_modules[0].get_source()

dump(mod.script(), IRModule_dir, "IRModule.py")
dump(best_cuda_program, CUDA_dir, "CUDA.cu")

func = tvm.build(sch, args, target)
a_np = np.random.uniform(size=(M, K)).astype(np.float32)
b_np = np.random.uniform(size=(K, N)).astype(np.float32)
out_np = a_np.dot(b_np)

dev = tvm.cuda(0)
a_tvm = tvm.nd.array(a_np, device=dev)
b_tvm = tvm.nd.array(b_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)
func(a_tvm, b_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
print("✅✅✅result correctness pass! ")

# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, dev, number=10, repeat=3)

_time = evaluator(a_tvm, b_tvm, out_tvm).mean
print("Execution time of the best schedule: %.3f ms" % (_time * 1000))
print("Tuned Kernel's TFLOPs: %.3f" % (2 * M * K * N / _time / 1e12))
print("✅✅✅benchmark pass! ")
