#!/usr/bin/env python3
import argparse
import math
import time

import flashinfer
import flashinfer.decode as fi_decode
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashInfer decode benchmark (full attention compute).")
    parser.add_argument("--mode", choices=("paged", "single"), default="paged")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seqlen-k", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--permute-blocks", type=int, default=1)
    parser.add_argument("--permute-a", type=int, default=5)
    parser.add_argument("--permute-b", type=int, default=3)
    parser.add_argument("--use-tensor-cores", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dtype", type=str, choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--workspace-mb", type=int, default=128)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def validate_args(args: argparse.Namespace) -> None:
    if args.batch <= 0:
        raise ValueError("--batch must be > 0")
    if args.seqlen_k <= 0:
        raise ValueError("--seqlen-k must be > 0")
    if args.num_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("--num-heads and --num-kv-heads must be > 0")
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads (GQA)")
    if args.head_dim <= 0:
        raise ValueError("--head-dim must be > 0")
    if args.page_size <= 0:
        raise ValueError("--page-size must be > 0")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup >= 0 and --iters > 0")
    if args.workspace_mb <= 0:
        raise ValueError("--workspace-mb must be > 0")


def build_paged_tables(
    batch: int,
    seqlen_k: int,
    page_size: int,
    permute_blocks: bool,
    permute_a: int,
    permute_b: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    pages_per_seq = (seqlen_k + page_size - 1) // page_size
    if pages_per_seq <= 0:
        raise ValueError("pages_per_seq must be > 0")

    indptr = torch.arange(
        0,
        (batch + 1) * pages_per_seq,
        pages_per_seq,
        device=device,
        dtype=torch.int32,
    )

    logical = torch.arange(pages_per_seq, device=device, dtype=torch.int32)
    per_seq_indices: list[torch.Tensor] = []
    for b in range(batch):
        lb = logical
        if permute_blocks:
            lb = (permute_a * lb + permute_b + b) % pages_per_seq
        # Interleave pages from different requests in a shared pool.
        phys = lb * batch + b
        per_seq_indices.append(phys.to(dtype=torch.int32))
    indices = torch.cat(per_seq_indices, dim=0)

    remainder = seqlen_k % page_size
    last_len = page_size if remainder == 0 else remainder
    last_page_len = torch.full((batch,), last_len, device=device, dtype=torch.int32)
    return indptr, indices, last_page_len, pages_per_seq


def run_paged_decode(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    use_tensor_cores = bool(args.use_tensor_cores)
    indptr, indices, last_page_len, pages_per_seq = build_paged_tables(
        batch=args.batch,
        seqlen_k=args.seqlen_k,
        page_size=args.page_size,
        permute_blocks=bool(args.permute_blocks),
        permute_a=args.permute_a,
        permute_b=args.permute_b,
        device=device,
    )

    max_num_pages = args.batch * pages_per_seq
    q = torch.randn(args.batch, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        max_num_pages,
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.randn(
        max_num_pages,
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )

    workspace = torch.empty(args.workspace_mb * 1024 * 1024, device=device, dtype=torch.uint8)
    wrapper = fi_decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_tensor_cores=use_tensor_cores,
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    out = None
    for _ in range(args.warmup):
        out = wrapper.run(q, (k_cache, v_cache))

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.iters):
        out = wrapper.run(q, (k_cache, v_cache))
    torch.cuda.synchronize()
    elapsed = time.time() - start

    if out is None:
        raise RuntimeError("decode output is empty")
    print(
        f"flashinfer decode done: mode=paged elapsed_s={elapsed:.6f} "
        f"shape={tuple(out.shape)} checksum={float(out.float().mean().item()):.6f}"
    )
    return out


def run_single_decode(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    use_tensor_cores = bool(args.use_tensor_cores)
    q = torch.randn(args.batch, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.seqlen_k, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.seqlen_k, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)

    out = None
    for _ in range(args.warmup):
        outs = []
        for b in range(args.batch):
            outs.append(
                fi_decode.single_decode_with_kv_cache(
                    q[b],
                    k[b],
                    v[b],
                    kv_layout="NHD",
                    use_tensor_cores=use_tensor_cores,
                )
            )
        out = torch.stack(outs, dim=0)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.iters):
        outs = []
        for b in range(args.batch):
            outs.append(
                fi_decode.single_decode_with_kv_cache(
                    q[b],
                    k[b],
                    v[b],
                    kv_layout="NHD",
                    use_tensor_cores=use_tensor_cores,
                )
            )
        out = torch.stack(outs, dim=0)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    if out is None:
        raise RuntimeError("decode output is empty")
    print(
        f"flashinfer decode done: mode=single elapsed_s={elapsed:.6f} "
        f"shape={tuple(out.shape)} checksum={float(out.float().mean().item()):.6f}"
    )
    return out


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    dtype = resolve_dtype(args.dtype)

    print(
        "flashinfer_decode config: "
        f"mode={args.mode} batch={args.batch} seqlen_k={args.seqlen_k} "
        f"heads={args.num_heads} kv_heads={args.num_kv_heads} head_dim={args.head_dim} "
        f"page_size={args.page_size} permute={args.permute_blocks} "
        f"dtype={args.dtype} "
        f"use_tensor_cores={args.use_tensor_cores} warmup={args.warmup} iters={args.iters}"
    )
    print(
        f"versions: torch={torch.__version__} cuda={torch.version.cuda} "
        f"flashinfer={flashinfer.__version__}"
    )

    if args.mode == "paged":
        run_paged_decode(args, dtype=dtype, device=device)
    elif args.mode == "single":
        run_single_decode(args, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
