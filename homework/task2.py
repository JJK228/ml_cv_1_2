import functools
import platform

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
    ],
    key=["N"],
)
@triton.jit
def _layernorm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight + bias

    tl.store(y_ptr + row * N + offsets, y, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
    ],
    key=["N"],
    reset_to_zero=["grad_weight_ptr", "grad_bias_ptr"],
)
@triton.jit
def _layernorm_backward_kernel(
    grad_y_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    grad_x_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    grad_y = tl.load(grad_y_ptr + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    x_hat = (x - mean) * rstd
    grad_x_hat = grad_y * weight
    grad_x_hat = tl.where(mask, grad_x_hat, 0.0)
    x_hat = tl.where(mask, x_hat, 0.0)

    sum_grad = tl.sum(grad_x_hat, axis=0)
    sum_grad_x_hat = tl.sum(grad_x_hat * x_hat, axis=0)
    grad_x = (grad_x_hat - sum_grad / N - x_hat * sum_grad_x_hat / N) * rstd

    tl.store(grad_x_ptr + row * N + offsets, grad_x, mask=mask)
    tl.atomic_add(grad_weight_ptr + offsets, grad_y * x_hat, sem="relaxed", mask=mask)
    tl.atomic_add(grad_bias_ptr + offsets, grad_y, sem="relaxed", mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


class _LayerNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        if not x.is_cuda:
            raise ValueError("Triton LayerNorm expects CUDA tensors")
        if weight.ndim != 1 or bias.ndim != 1 or weight.shape != bias.shape:
            raise ValueError("weight and bias must be 1D tensors with the same shape")
        if x.shape[-1] != weight.numel():
            raise ValueError("x.shape[-1] must match weight and bias size")

        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1])
        weight = weight.contiguous()
        bias = bias.contiguous()
        M, N = x_2d.shape
        if _next_power_of_2(N) > 2048:
            raise ValueError("This educational kernel supports hidden size up to 2048")

        y = torch.empty_like(x_2d)
        mean = torch.empty((M,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

        _layernorm_forward_kernel[(M,)](
            x_2d,
            weight,
            bias,
            y,
            mean,
            rstd,
            N=N,
            eps=eps,
        )

        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.original_shape = original_shape
        ctx.N = N
        return y.view(original_shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        grad_y = grad_y.contiguous().view(-1, ctx.N)
        M, N = grad_y.shape

        grad_x = torch.empty_like(x)
        grad_weight = torch.zeros((N,), device=x.device, dtype=torch.float32)
        grad_bias = torch.zeros((N,), device=x.device, dtype=torch.float32)

        _layernorm_backward_kernel[(M,)](
            grad_y,
            x,
            weight,
            mean,
            rstd,
            grad_x,
            grad_weight,
            grad_bias,
            N=N,
        )

        return (
            grad_x.view(ctx.original_shape),
            grad_weight.to(weight.dtype),
            grad_bias.to(weight.dtype),
            None,
        )


def layernorm_forward_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    return x_hat * weight + bias


def layernorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return _LayerNormTriton.apply(x, weight, bias, eps)


def check_correctness() -> None:
    torch.manual_seed(0)
    device = "cuda"
    shapes = [(16, 64), (32, 128), (8, 17, 256), (4, 9, 513)]

    for shape in shapes:
        x = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(shape[-1], device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(shape[-1], device=device, dtype=torch.float32, requires_grad=True)
        grad = torch.randn_like(x)

        y_torch = layernorm_forward_torch(x, weight, bias)
        y_triton = layernorm_triton(x, weight, bias)
        torch.testing.assert_close(y_triton, y_torch, rtol=1e-5, atol=1e-5)

        y_torch.backward(grad, retain_graph=True)
        grads_torch = (x.grad.detach().clone(), weight.grad.detach().clone(), bias.grad.detach().clone())
        x.grad = None
        weight.grad = None
        bias.grad = None

        y_triton.backward(grad)
        torch.testing.assert_close(x.grad, grads_torch[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(weight.grad, grads_torch[1], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(bias.grad, grads_torch[2], rtol=1e-4, atol=1e-4)

    print("correctness: ok")


def run_benchmark() -> None:
    providers = {
        "triton": layernorm_triton,
        "torch": layernorm_forward_torch,
    }

    @triton.testing.perf_report(
        [
            triton.testing.Benchmark(
                x_names=["hidden_size"],
                x_vals=[64, 128, 256, 512, 1024, 2048],
                line_arg="provider",
                line_vals=list(providers.keys()),
                line_names=list(providers.keys()),
                styles=[("blue", "-"), ("red", "--"), ("green", "--")],
                ylabel="GB/s",
                plot_name="layernorm_forward_fp32",
                args={"M": 8192},
            )
        ]
    )
    def benchmark(M: int, hidden_size: int, provider: str):
        x = torch.randn((M, hidden_size), device="cuda", dtype=torch.float32)
        weight = torch.randn((hidden_size,), device="cuda", dtype=torch.float32)
        bias = torch.randn((hidden_size,), device="cuda", dtype=torch.float32)
        fn = functools.partial(providers[provider], x, weight, bias)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

        def gbps(time_ms: float) -> float:
            total_bytes = 4 * M * hidden_size * x.element_size()
            return (total_bytes * 1e-9) / (time_ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(save_path=None, show_plots=False, print_data=True)


if __name__ == "__main__":
    if platform.system() == "Windows":
        raise RuntimeError(
            "Run this Triton homework from WSL/Linux, not Windows PowerShell:\n"
            "wsl.exe -- bash -lc \"cd /mnt/c/Users/baslo/ml_cv_1_2 && "
            "CC=~/.local/bin/zigcc python3 homework/task2.py\""
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton homework")
    check_correctness()
    run_benchmark()
