import argparse
import torch
import flash_attn
from flash_attn import flash_attn_func


def test_flash_attention(batch_size, seqlen, nheads, kv_heads, head_dim):
    print(f"PyTorch: {torch.__version__}, flash_attn: {flash_attn.__version__}")
    print(f"Config: B={batch_size} S={seqlen} H={nheads} kv_H={kv_heads} D={head_dim} causal=True dtype=bf16")

    device = "cuda"
    dtype = torch.bfloat16

    # Q has H heads, K/V have kv_heads (MHA when kv_H==H, else GQA)
    q = torch.randn(batch_size, seqlen, nheads,    head_dim, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen, kv_heads,  head_dim, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen, kv_heads,  head_dim, device=device, dtype=dtype, requires_grad=False)

    out = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    print(f"OK. output={tuple(out.shape)} dtype={out.dtype}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--nheads", type=int, default=32)
    p.add_argument("--kv_heads", type=int, default=32, help="GQA: kv_heads<nheads; MHA: kv_heads==nheads")
    p.add_argument("--d", type=int, default=128)
    args = p.parse_args()
    test_flash_attention(args.batch_size, args.seqlen, args.nheads, args.kv_heads, args.d)
