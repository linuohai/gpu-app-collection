import torch
import flash_attn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

def test_flash_attention(batch_size, seqlen, nheads, d):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Flash Attention version: {flash_attn.__version__}")

    device = "cuda"
    dtype = torch.float16
    
    # 设置参数
    print(f"Configuration: batch_size={batch_size}, seqlen={seqlen}, nheads={nheads}, d={d}")
    
    # 创建输入数据 (Batch, Seqlen, 3, nheads, headdim)
    # 不需要梯度 (requires_grad=False)，因为我们只关心前向传播的访存
    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=False)
    
    print("\nRunning Flash Attention forward pass...")
    try:
        # 运行 Flash Attention
        # 这里的 out = ... 就是核心算子执行的地方
        # 模拟器或 Profiler 会在这里捕获内存访问
        out = flash_attn_qkvpacked_func(qkv)
        
        print("Forward pass successful!")
        print(f"Output shape: {out.shape}")
        
        # 强制同步，确保 GPU 任务执行完成
        torch.cuda.synchronize()
        
    except Exception as e:
        print(f"Error running Flash Attention: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Flash Attention")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seqlen", type=int, default=128, help="Sequence length")
    parser.add_argument("--nheads", type=int, default=8, help="Number of heads")
    parser.add_argument("--d", type=int, default=64, help="Head dimension")
    
    args = parser.parse_args()
    
    test_flash_attention(args.batch_size, args.seqlen, args.nheads, args.d)