import math

import torch
import triton
import triton.language as tl


# forward; backward

# 1d-grid

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["N_bucket"]
)
@triton.jit
def silu_mul_forward_k(
        x_ptr: tl.pointer_type,
        y_ptr: tl.pointer_type,
        z_ptr: tl.pointer_type,

        N: int,
        N_bucket: int,  # autotune

        BLOCK_SIZE: tl.constexpr
):
    #  0    1   2 ...
    # [---|---|---|...

    pid = tl.program_id(axis=0)

    # [---|---|---|...
    # /\  /\  /\
    start_idx = pid * BLOCK_SIZE

    # [0..31], [32..63], [64..95]
    block_offsets = start_idx + tl.arange(0, BLOCK_SIZE)

    block_x_ptr = x_ptr + block_offsets
    block_y_ptr = y_ptr + block_offsets
    block_z_ptr = z_ptr + block_offsets

    block_mask = block_offsets < N

    block_x = tl.load(block_x_ptr, mask=block_mask)
    block_y = tl.load(block_y_ptr, mask=block_mask)

    # silu mul: silu(x)*y = sigmoid(x)*x*y
    block_result = tl.sigmoid(block_x.to(tl.float32)) * block_x * block_y

    tl.store(block_z_ptr, block_result, mask=block_mask)


def bucketize_n(n: int) -> int:
    return int(math.log(n))


def silu_mul_forward(
        x: torch.Tensor,
        y: torch.Tensor
):
    if x.shape != y.shape:
        raise ValueError()
    if x.device != y.device:
        raise ValueError()

    if not x.is_contiguous():
        x = x.contiguous()

    if not y.is_contiguous():
        y = y.contiguous()

    N = x.numel()
    n_bucket = bucketize_n(N)

    z = torch.empty_like(x)

    def grid_fn(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    silu_mul_forward_k[grid_fn](
        x,
        y,
        z,
        N=N,
        N_bucket=n_bucket
    )

    return z

if __name__ == '__main__':
    # [batch; seq len; hidden size]
    x = torch.randn((12, 81, 199), device="cuda").bfloat16()
    y = torch.randn((12, 81, 199), device="cuda").bfloat16()

    result_torch = torch.nn.functional.silu(x) * y

    result_triton = silu_mul_forward(x, y)

    torch.testing.assert_close(
        result_torch,
        result_triton
    )
