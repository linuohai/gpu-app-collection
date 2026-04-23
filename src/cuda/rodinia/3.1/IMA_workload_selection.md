# IMA 工作负载（Rodinia 3.1）输入选择：`bfs` / `b+tree`

目标：在 Rodinia 3.1 中挑出 **IMA（Indirect Memory Access，间接访存）**最典型、且在 Accel-Sim trace/缓存预取研究里“可控、可复现”的 workload，并给出推荐输入数据集（含理由）。这里的 IMA 主要指 `A[idx]` 中 `idx` 来自内存/数据结构（如 `neighbor_id`、子节点指针/索引）的 **data-dependent address**。

> 对比 Gardenia：Gardenia 更偏“真实世界图”（幂律/Hub/稀疏矩阵），Rodinia 3.1 里的图类 workload 主要只有 BFS，且自带输入更偏合成随机图；因此 **Rodinia 的价值**更多是提供“非常干净的 worst-case IMA 基线”（随机邻居、低局部性），便于验证/对比 cache prefetch 的上限与副作用。

本文件对应 `util/job_launching/apps/define-all-apps.yml` 中 `rodinia-3.1` 套件（`exec_dir=$GPUAPPS_ROOT/bin/$CUDA_VERSION/release/`，`data_dirs=$GPUAPPS_ROOT/data_dirs/cuda/rodinia/3.1/`）。

---

## TL;DR（Rodinia 里“最像 IMA”的两个代表）

- **图 IMA（邻居索引数组）**：`bfs-rodinia-3.1`
  - 推荐输入：`graph1MW_6.txt`（最强 IMA/最大工作集）；或 `graph65536.txt`（更可控）
- **指针追踪 IMA（树遍历）**：`b+tree-rodinia-3.1`
  - 推荐输入：`mil.txt` + `command.txt`（1M keys + 批量查询）

Rodinia 3.1 的其他 benchmark（如 stencil/图像处理/密集计算）主要是规则访存或计算受限，不适合作为 IMA 预取/trace 分析的“代表 workload”；`huffman/mummergpu` 等虽有查表/间接访问成分，但在本仓库的 `rodinia-3.1` suite 配置里默认未启用（见 `define-all-apps.yml` 注释项）。

---

## 1) `bfs-rodinia-3.1`（图遍历 IMA：`visited[neighbor]` / `cost[neighbor]`）

### 1.1 算法与 IMA 热点

Rodinia BFS 是典型的 **frontier-based BFS**（两段 kernel 循环直到 frontier 为空）：

- `Kernel`：对当前 frontier 中的顶点 `tid`，扫描其邻居边 `g_graph_edges[i]` 得到 `id`，然后访问：
  - `g_graph_visited[id]`（读）
  - `g_cost[id]`（写）
  - `g_updating_graph_mask[id]`（写）
- 这些都是典型 IMA：地址由 `id = g_graph_edges[i]` 决定。

代码位置：

- IMA 核心：`gpu-app-collection/src/cuda/rodinia/3.1/cuda/bfs/kernel.cu`
- 注意：`bfs.cu` 虽然读入 `source`，但随后强制 `source=0`，因此**无法通过 source 调参**（只能靠输入图本身）。

### 1.2 输入格式（Rodinia BFS graph*.txt）

输入文件是 CSR 变体：

1. `no_of_nodes`
2. 对每个顶点 `i`：`starting_i  no_of_edges_i`
3. `source`（但程序会覆盖为 0）
4. `edge_list_size`
5. `edge_list_size` 行：`dst_id  cost`（cost 读入但 BFS 实际只用 `dst_id`）

### 1.3 自带数据集概览（以及“为什么它很 IMA”）

数据集位置：`gpu-app-collection/data_dirs/cuda/rodinia/3.1/bfs-rodinia-3.1/data/`

这三张图基本都是 **平均度≈6、邻居 id 近似均匀随机覆盖 [0, |V|)** 的合成图。其效果是：

- 对于 `visited[neighbor] / cost[neighbor]` 这类“邻居 id 做索引”的数组访问，**同一顶点的邻居往往落在不同 cache line**（最典型 IMA）。
- 邻居跨度接近 `|V|`，对 L1/L2/TLB 都更不友好。

下面统计假设 128B cache line、`cost[]` 为 4B 元素（line_elems=32），并给出每个顶点邻居列表的“邻居/line”指标（越接近 1 越随机、越 IMA）：

| dataset | \|V\| | \|E\| | BFS depth(src=0) | 每点邻居/line（均值 / p50） | 邻居跨度 p50（越大越随机） | 设备侧大致 footprint（核心数组） |
|---|---:|---:|---:|---:|---:|---:|
| `graph4096.txt` | 4,096 | 24,576 | 7 | 1.02 / 1.00 | 2,965 | ~0.15 MB |
| `graph65536.txt` | 65,536 | 393,216 | 9 | 1.00 / 1.00 | 47,175 | ~2.5 MB |
| `graph1MW_6.txt` | 1,000,000 | 5,999,970 | 11 | 1.23 / 1.20 | 667,966 | ~39 MB |

> footprint 估算（device）：`Node[|V|]`(8B) + `edges[|E|]`(4B) + `cost[|V|]`(4B) + `mask/visited/updating`(≈3×1B×|V|)；不含运行时额外开销。

### 1.4 最终推荐（BFS）

**推荐 1（优先“强 IMA + 大工作集”，用于评估 L2/HBM/预取收益上限）：`graph1MW_6.txt`**

- 工作集显著大于典型 L2（约 39MB 级），能把 `visited/cost` 的随机访问推向 L2 miss/HBM，IMA 行为更“纯粹”。
- 邻居跨度大、每点邻居/line 接近 1（接近“一邻居一条 cache line”的最坏情形），对预取/缓存策略更敏感。

**推荐 2（优先“更可控、更快迭代”，仍然很 IMA）：`graph65536.txt`**

- 每点邻居/line≈1.0（随机性非常强），但 footprint 只有 ~2.5MB，更容易控制仿真时间与 trace 体量。
- 适合作为“打开 issue/L1/L2 trace 也能跑得动”的调参与回归基线。

`graph4096.txt` 建议仅用于“流程/功能跑通”，不适合做 cache prefetch 结论（太小，容易被缓存/启动开销淹没）。

---

## 2) `b+tree-rodinia-3.1`（指针追踪 IMA：树遍历 + 记录索引）

### 2.1 算法与 IMA 热点

Rodinia 的 B+Tree benchmark 会把输入 key 插入树中，然后按命令文件执行批量查询。GPU kernel 中典型 IMA：

- **树遍历（pointer chasing）**：下一层节点索引来自当前节点的 `indices[]`，属于 data-dependent address。
- **记录访问**：`recordsD[ knodesD[leaf].indices[i] ]` 也是间接索引。

代码位置：

- 关键 kernel：`gpu-app-collection/src/cuda/rodinia/3.1/cuda/b+tree/kernel/kernel_gpu_cuda.cu`（`currKnodeD[...]` / `offsetD[...]` 更新与 `recordsD[...]` 访问）

与 BFS 不同：B+Tree 的访问具有更强的 **串行依赖链**（每层遍历依赖上一层结果），更适合观察“cache miss latency + MLP + 预取准确率/污染”的权衡。

### 2.2 输入与命令格式

数据集位置：`gpu-app-collection/data_dirs/cuda/rodinia/3.1/b+tree-rodinia-3.1/data/`

- `mil.txt`：首行是 `N`，后续 `N` 行是 key（Rodinia 默认是 `0..N-1`）。
- `command.txt`：脚本化命令序列（非交互）。常用两类：
  - `k <count>`：随机点查询（bundled queries）
  - `j <count> <rSize>`：随机范围查询（range search）

### 2.3 最终推荐（B+Tree）

**推荐：`mil.txt` + `command.txt`（默认 1M keys + 混合查询）**

- `N=1,000,000` 使得叶子/记录集合足够大，查询 key（尤其 `k` 随机点查）会把访问分散到大量叶节点与记录，形成典型 IMA。
- 顶层节点会形成热点（较容易被缓存命中），叶层/记录更随机：对“预取是否能拉近叶层 miss”的分析很直观。

如果你的目标是“更纯粹的 IMA（减少顺序扫描带来的局部性）”，可把 `command.txt` 改成只保留较大的 `k`（例如 `k 100000`）；如果你想研究“IMA + 顺序段混合”的场景，则保留 `j`（会引入更强的 leaf-level 顺序性）。

---

## 3) 建议的运行命令（与 job launcher 一致）

以下命令行参数与 `util/job_launching/apps/define-all-apps.yml` 中 `rodinia-3.1` 的写法一致；其中 `./data/...` 由 job launcher 在运行目录下映射到对应 benchmark 的数据集目录。

### BFS

```bash
./bfs-rodinia-3.1 ./data/graph1MW_6.txt
```

### B+Tree

```bash
./b+tree-rodinia-3.1 file ./data/mil.txt command ./data/command.txt
```
