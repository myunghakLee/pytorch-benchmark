"""
A lightweight runner that just sets up a model and runs one of its functions in a particular configuration.

Intented for debugging/exploration/profiling use cases, where the test/measurement harness is overhead.

DANGER: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
        has been installed.  This script intentionally does not automate or enforce setup steps.

Wall time provided for sanity but is not a sane benchmark measurement.
"""
import argparse
import time
import multiprocessing
from datetime import datetime
import csv
import torch.autograd.profiler as profiler
from py3nvml import py3nvml
from torchbenchmark import list_models

import torch


def device_power_profiler(connection):
    py3nvml.nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    header = ["elapsed_time", "power"]
    with open("/workspace/benchmark/ncu_reports/power_output.csv", "w") as f:
        print("power profiler ready")
        writer = csv.writer(f)
        writer.writerow(header)
        flag = connection.recv()
        t0 = time.time()
        while not connection.poll():
            writer.writerow([str(time.time() - t0), nvmlDeviceGetPowerUsage(handle)])



def run_one_step(func):

    func()

    if args.device == "cuda":
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        
        t0 = time.time()
        func()
        t1 = time.time()

        end_event.record()
        torch.cuda.synchronize()
        print(f"Ran in {t1 - t0} seconds, gpu time {start_event.elapsed_time(end_event)}.")

    else:

        t0 = time.time()
        func()
        t1 = time.time()

        print(f"Ran in {t1 - t0} seconds.")


def profile_one_step(func, nwarmup=3):
    for i in range(nwarmup):
        func()

    use_cuda = args.device == "cuda"
    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(target=device_power_profiler, args=(child_conn, ))
    proc.start()
    print("power profiler kicked off")
    parent_conn.send("start")
    with profiler.profile(record_shapes=True, use_cuda = use_cuda) as prof:
        func()

    parent_conn.send("stop")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", help="Full or partial name of a model to run.  If partial, picks the first match.")
    parser.add_argument("-d", "--device", choices=["cpu",  "cuda"], default="cpu", help="Which device to use.")
    parser.add_argument("-m", "--mode", choices=["eager",  "jit"], default="eager", help="Which mode to run.")
    parser.add_argument("-t", "--test", choices=["eval",  "train"], default="eval", help="Which test to run.")
    parser.add_argument("--profile", action="store_true", help="Run the profiler around the function")
    args = parser.parse_args()
    found = False
    for Model in list_models():
        if args.model.lower() in Model.name.lower():
            found = True
            break
    if found:
        print(f"Running {args.test} method from {Model.name} on {args.device} in {args.mode} mode")
    else:
        print(f"Unable to find model matching {args.model}")
        exit(-1)

    # build the model and get the chosen test method
    m = Model(device = args.device, jit = (args.mode == "jit"))
    test = getattr(m, args.test)

    if args.profile:
        profile_one_step(test)
    else:
        run_one_step(test)
