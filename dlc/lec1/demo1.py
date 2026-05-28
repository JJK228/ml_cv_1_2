from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

if __name__ == '__main__':
    a = torch.randn(10000, 10000, device="cuda")
    b = torch.randn(10000, 10000, device="cuda")

    print("Warming up...")
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    print("Profiling starts...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True, with_stack=True) as prof:
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        # independent operation A
        with torch.cuda.stream(stream1):
            res1_stream = torch.matmul(a, b)

        # independent operation B
        with torch.cuda.stream(stream2):
            res2_stream = torch.matmul(a, b)

    trace_filename = Path("matmul_trace.json")
    prof.export_chrome_trace(str(trace_filename))
    print(f"\nTrace was saved to: {trace_filename.absolute()}")
