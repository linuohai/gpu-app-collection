# FlashInfer Decode 基准（完整算子）

这个目录提供的是 **完整 decode attention** 基准，不是“只模拟访存模式”的微内核。  
核心调用来自 FlashInfer：

- `flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper`（paged KV，贴近 vLLM 场景）
- `flashinfer.decode.single_decode_with_kv_cache`（single-request decode，对照模式）

默认 wrapper 走 paged 模式，并提供两组序列长度：

- `flashinfer_decode_512`
- `flashinfer_decode_4096`

## 1. 为什么这是“完整 decode 算子”

`flashinfer_decode.py` 不是手写简化循环，而是直接调用 FlashInfer decode API。  
也就是说，内核内部包含真实 decode attention 计算流程（包括 logits、softmax、对 V 的加权归约），不是仅做 K/V 读取累加。

## 2. 代码逻辑（通俗版）

文件：`flashinfer_decode.py`

程序流程可以理解成 5 步：

1. 读参数并校验  
   例如 `num_heads % num_kv_heads == 0`，保证 GQA 分组合法。
2. 构造输入  
   - `q`: `[batch, num_heads, head_dim]`  
   - paged `k_cache/v_cache`: `[max_pages, page_size, num_kv_heads, head_dim]`
3. 构造 paged 索引表  
   - `indptr`: 每个请求的 page 段边界  
   - `indices`: 逻辑 page -> 物理 page 的映射  
   - `last_page_len`: 每个请求最后一页的有效 token 数
4. `wrapper.plan(...)`  
   FlashInfer 在这里为当前问题规模准备执行计划。
5. `wrapper.run(...)`  
   执行完整 decode attention，输出 `out`，并打印形状和 checksum。

## 3. 一个具体例子（paged 索引）

假设：

- `batch=2`
- `seqlen_k=64`
- `page_size=16`

则每个请求有 `pages_per_seq=4` 页。  
如果开启 `--permute-blocks 1`，每个请求的逻辑页会按公式置换后再映射到共享物理页池。  
直观上：模型“看起来”在读第 0/1/2/3 页，但物理上可能跳到较远地址，符合线上 paged KV 的不规则访问特征。

## 4. 构建

先确保 FlashInfer 可导入（若缺失可安装）：

```bash
python3 -m pip install --no-cache-dir flashinfer-python==0.6.2
```

然后确认版本：

```bash
python3 - <<'PY'
import flashinfer, torch
print("flashinfer", flashinfer.__version__, "torch", torch.__version__)
PY
```

在 `gpu-app-collection/src` 下执行：

```bash
source setup_environment
make -j flashinfer_decode
```

会把以下文件拷贝到 `$GPUAPPS_ROOT/bin/$CUDA_VERSION/release/`：

- `flashinfer_decode.py`
- `flashinfer_decode_512`
- `flashinfer_decode_4096`

## 5. 直接运行（不走 trace）

```bash
$GPUAPPS_ROOT/bin/$CUDA_VERSION/release/flashinfer_decode_512
$GPUAPPS_ROOT/bin/$CUDA_VERSION/release/flashinfer_decode_4096
```

可以看到：

- 版本信息（torch/cuda/flashinfer）
- 配置参数
- 输出张量 shape 与 checksum

## 6. 生成 trace

已经注册到 `define-all-apps.yml` 的 suite `flashinfer_decode`，可直接：

```bash
cd accel-sim-framework/util/tracer_nvbit
./run_hw_trace.py -B flashinfer_decode -D 0
```

产物路径示例：

```text
accel-sim-framework/hw_run/traces/device-0/<CUDA_VERSION>/flashinfer_decode_4096/NO_ARGS/traces/kernelslist.g
```

### 6.1 只保留 decode kernel（建议）

PyTorch + FlashInfer 会触发很多准备阶段 kernel（随机数、张量变换等）。  
如果你只想在 Accel-Sim 中研究 decode attention，本质上只需要保留 FlashInfer decode 对应的 kernel。

定位方法：

```bash
cd accel-sim-framework/hw_run/traces/device-0/<CUDA_VERSION>/flashinfer_decode_512/NO_ARGS/traces
grep -n "flashinfer" stats_ctx_*
```

在一次实际运行中，可见 decode 相关是：

- `kernel-56`：`BatchPrefillWithPagedKVCacheKernel...`
- `kernel-57`：`PersistentVariableLengthMergeStatesKernel...`

然后把 `kernelslist.g` 改成只保留这两个 `traceg.xz`（或你本次运行识别出的对应编号）。  
注意：不同环境或库版本，kernel 编号可能变化，建议每次都用 `stats_ctx_*` 重新确认。

## 7. Accel-Sim 运行

`traceL1` 中新增：

- `fi_dec_512`
- `fi_dec_4k`

示例：

```bash
cd accel-sim-framework
./traceL1 -gto --no-issue-trace fi_dec_512 fi_dec_512_base
./traceL1 -gto --ideal-l1d --no-issue-trace fi_dec_512 fi_dec_512_ideal_l1d
```

批量脚本：

```bash
cd accel-sim-framework
./run_flashinfer_decode_compare.sh --no-issue-trace --suffix run1
```

## 8. 常用参数建议

- 快速迭代：先跑 `512`
- 长上下文压测：跑 `4096`
- 想让跨页更不规则：保持 `--permute-blocks 1`
- 想缩短仿真时间：在 `traceL1` 加 `--max-completed-cta` 或 `--max-cycle`

## 9. 当前参数大概对应什么级别

默认 wrapper（`flashinfer_decode_4096`）配置是：

- `batch=8`
- `seqlen_k=4096`
- `num_heads=32`
- `num_kv_heads=8`
- `head_dim=128`
- `page_size=16`

这组头部几何（`32/8/128`）接近常见 7B~8B 级别模型单层 attention（GQA）：

- `hidden_size ≈ num_heads * head_dim = 32 * 128 = 4096`
- `q_per_kv = num_heads / num_kv_heads = 4`（每个 KV 头被 4 个 Q 头共享）

需要注意：这里是“单层 decode attention 子问题”，不是完整 32 层模型端到端推理。

### 一个直观量级例子

按 fp16 粗略估算单层 KV cache 占用：

- 每 token KV 字节数 = `2(K+V) * num_kv_heads * head_dim * 2B`
- 代入默认参数：`2 * 8 * 128 * 2 = 4096B = 4KB/token`
- 单层总 KV（`batch=8, seqlen=4096`）约 `8 * 4096 * 4KB = 128MB`
- 如果按 32 层做量级感知，约 `128MB * 32 = 4GB`

## 10. 每个参数是什么意思（通俗版）

- `--mode`：decode 形态。`paged` 表示 KV 像“分页文件柜”一样存，不是整块连续内存。
- `--batch`：并发请求数。`batch=8` 可以理解为“同时服务 8 个用户”。
- `--seqlen-k`：历史上下文长度。`4096` 表示每个请求要回看 4096 个 token。
- `--num-heads`：Q/O 头数量。像 32 组并行“注意力小组”。
- `--num-kv-heads`：K/V 头数量。用于 GQA 共享，`32/8=4` 表示 4 个 Q 头共享 1 个 KV 头。
- `--head-dim`：每个头内部向量长度。`128` 可以理解为每个头看 128 维特征。
- `--page-size`：每页容纳 token 数。`16` 表示每 16 个 token 一页。
- `--permute-blocks`：是否打乱逻辑页到物理页映射。`1` 更接近线上碎片化 paged KV。
- `--permute-a/--permute-b`：页映射置换参数，控制“跳页”模式。
- `--use-tensor-cores`：是否走 Tensor Core 路径。
- `--iters`：执行 decode 的步数（用于控制 trace 体量/运行时间）。
- `--workspace-mb`：FlashInfer 运行时临时工作区大小。

## 11. 如果要改参数，建议怎么改

先明确目标，再动参数。下面是常用方向：

### 方向 A：快速迭代（先看趋势）

- 把 `seqlen_k` 降到 `512`
- `batch` 设 `4` 或 `8`
- 保持 `num_heads=32, num_kv_heads=8, head_dim=128`

作用：更快出 trace，适合先验证“是否有提升趋势”。

### 方向 B：更贴近线上压力

- 提高 `seqlen_k` 到 `8192` 或 `16384`
- 提高 `batch` 到 `16` / `32`
- 保持 `mode=paged` 且 `permute-blocks=1`

作用：更接近在线长上下文和高并发 decode。

### 方向 C：放大 paged 不规则访存

- 减小 `page_size`（如 `16 -> 8`）
- 保持 `permute-blocks=1`

作用：跨页更频繁、地址更不连续，更容易暴露 cache/TLB 问题。

### 方向 D：对照实验（定位根因）

- 固定所有参数，每次只改一个维度
  - 先改 `seqlen_k`
  - 再改 `batch`
  - 再改 `page_size`

作用：可以更清楚归因“哪一项参数导致瓶颈变化”。
