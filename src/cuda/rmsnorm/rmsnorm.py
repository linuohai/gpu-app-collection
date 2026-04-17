import argparse
import torch


def test_rmsnorm(M, hidden, impl="vllm"):
    device = "cuda"
    dtype = torch.bfloat16

    print(f"RMSNorm M={M} hidden={hidden} impl={impl} dtype=bf16")

    x = torch.randn(M, hidden, dtype=dtype, device=device)
    residual = torch.randn(M, hidden, dtype=dtype, device=device)

    if impl == "vllm":
        from vllm.model_executor.layers.layernorm import RMSNorm
        layer = RMSNorm(hidden_size=hidden, eps=1e-6).to(device, dtype)
        # warmup
        _ = layer(x.clone(), residual=residual.clone())
        torch.cuda.synchronize()
        # benchmark
        out, new_res = layer(x, residual=residual)
    elif impl == "flash_attn":
        from flash_attn.ops.rms_norm import rms_norm
        weight = torch.ones(hidden, dtype=dtype, device=device)
        # warmup
        _ = rms_norm(x + residual, weight, 1e-6)
        torch.cuda.synchronize()
        # benchmark: residual add + rmsnorm fused manually
        added = x + residual
        out = rms_norm(added, weight, 1e-6)
    else:
        raise ValueError(f"unknown impl {impl}")

    torch.cuda.synchronize()
    print(f"OK. output={tuple(out.shape)} dtype={out.dtype}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--impl", type=str, default="vllm", choices=["vllm", "flash_attn"])
    args = p.parse_args()
    test_rmsnorm(args.M, args.hidden, args.impl)
