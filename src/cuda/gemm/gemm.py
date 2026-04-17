import argparse
import torch


def test_gemm(M, K, N, dtype_str="bf16"):
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    device = "cuda"

    print(f"GEMM [M={M}, K={K}] @ [K={K}, N={N}] dtype={dtype}")

    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

    # warmup
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # benchmark call (NVBit 会 trace 这里)
    out = torch.matmul(a, b)
    torch.cuda.synchronize()

    print(f"OK. output={tuple(out.shape)} dtype={out.dtype}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    args = p.parse_args()
    test_gemm(args.M, args.K, args.N, args.dtype)
