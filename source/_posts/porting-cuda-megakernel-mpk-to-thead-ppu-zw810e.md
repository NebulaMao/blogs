---
title: "Porting NVIDIA's CUDA Megakernel (MPK) to China's T-Head PPU-ZW810E"
date: 2026-06-27 12:00:00
updated: 2026-06-27 12:00:00
tags:
  - CUDA
  - MPK
  - 国产GPU
  - 平头哥PPU
  - LLM推理
  - 性能调优
categories:
  - AI工程
description: 把专为 NVIDIA Hopper/Blackwell 手写的 Mirage Persistent Kernel(MPK) megakernel 移植到阿里平头哥真武 810E(PPU-ZW810E)国产卡——不仅跑通、算对，单卡 Qwen3-8B 推理还反超了 PPU 自家优化栈。完整的踩坑、破案与调优全记录，所有数据来自真机。
cover: /images/mpk-ppu/01-ppu-smi-16cards.png
---

> 一个"理论上不可能"的实验：把专为 NVIDIA Hopper/Blackwell 手写的 Mirage Persistent Kernel(MPK)megakernel，跑到阿里平头哥的真武 810E(PPU-ZW810E，非 NVIDIA 架构)上——结果不仅跑通，单卡 Qwen3-8B 推理还**反超了 PPU 自家优化栈**。
>
> 这篇记录了完整的踩坑、破案和调优过程。所有数据来自真机。

---

## TL;DR

- ✅ MPK 从源码在 PPU 上构建成功(CUDA 兼容,`__CUDA_ARCH__=800`)。
- ✅ Qwen3-8B megakernel **跑通且输出正确**。
- ✅ 修一个死锁后,**16 worker 单核路径 = 35.5 ms/token,反超 native 的 39.4 ms(1.11×)**。
- 🔬 用 PPU 全套调试工具(ppu-gdb / racecheck / synccheck / 占用探针 / 插桩)把一个高 worker 数的死锁逐层定性为 scheduler 规模化 livelock。

![16× PPU-ZW810E 总览](/images/mpk-ppu/01-ppu-smi-16cards.png)

*配图 1：16 张 PPU-ZW810E 国产卡一字排开 —— 全文的「主角合影」。*

---

## 一、主角:一个 CUDA 巨核,和一张不是 N 卡的卡

**MPK(Mirage Persistent Kernel)** 的思路很激进:把整个 LLM 推理的所有算子和通信,融进**一次 kernel launch** 的"巨型常驻核",靠核内调度器把任务分发给常驻的 worker。它是为 NVIDIA 手写的——满是 `cp.async`、`wgmma`、TMA、nvshmem 这些 N 卡专属指令。

**PPU-ZW810E(真武 810E)** 是阿里平头哥的自研 AI 芯片,**不是 NVIDIA 架构**,自研 PPU 并行架构 + ICN 片间互联。

把前者搬到后者上,第一反应是"架构不通"。但真相反转了。

---

## 二、第一个惊喜:PPU 其实"会说 CUDA"

PPU 的 SDK 是**CUDA 源码兼容**的:`ppu-clang`(clang/llvm)能编 CUDA C/C++,带 nvcc 包装,甚至支持 Triton。最关键的一行实测:

```
__CUDA_ARCH__ == 800
```

PPU 自报 **sm_80(Ampere)兼容**!这意味着 MPK 里所有 `#if __CUDA_ARCH__ >= 800` 的 sm80 代码路径**原样激活**——TMA/wgmma 这些 Hopper/Blackwell 专属的它没有,但 ampere 路径根本不需要。

![PPU 自报 __CUDA_ARCH__=800](/images/mpk-ppu/02-cuda-arch-800.png)

*配图 2：一行模板探针把 arch 值打出来 —— 输出里的 `A<800>` 直接证明 PPU 把自己当成 sm_80。*

---

## 三、构建地狱:全程国内镜像源

PPU 在阿里云机房,墙外的源基本都连不上。构建 MPK 踩了一连串坑,每个都得换国内镜像:

- **pip**:容器默认指向 T-Head 内部源,公共包没有 → 加 `-i https://mirrors.aliyun.com/pypi/simple/`。
- **Rust**:`setup.py` 缺 cargo 时跑 `curl https://sh.rustup.rs | sh`,卡死在 `info: downloading installer` → 换 aliyun rustup 镜像 + aliyun crates 镜像。
- **cutlass/json 子模块**:只传 include 不够,cmake 要 `CUDA.cmake`/`CMakeLists.txt`。
- **transformers**:容器是 5.3,demo 按 pin 的 4.57.1 写,报 `pad_token_id` → 降版本。

熬过去之后,`import mirage` 成功,`PersistentKernel` API 就位。

![MPK 构建成功 + import 验证](/images/mpk-ppu/03-import-ok.png)

*配图 3：MPK 在国产卡上从源码编译并加载成功 —— `has PersistentKernel: True` / `has get_configurations_from_gpu: True`。*

---

## 四、第一次跑通:native 基线

先不上 megakernel,用 demo 自带的 torch 路径(走 PPU 自家优化库)跑 Qwen3-8B:

```
Prompt length 39, generate length 256, per-token latency 39.444 ms
```

输出是一段连贯正确的"大语言模型简介"。**国产卡上,Qwen3-8B 正确推理,39.4 ms/token,基线确立。**

![native 推理输出 + 延迟](/images/mpk-ppu/04-native-output.png)

*配图 4：native(torch + PPU 优化库)跑出连贯正确的英文生成 + 最后一行 `per-token latency 39.444 ms` 基线。*

---

## 五、上 megakernel:编译通过、跑出正确结果

`--use-mirage` 一开,MPK 在 PPU 上**生成并编译了整个 14000+ 任务的巨核**:

```
[MPK INIT] Total tasks: 14140, Total events: 2246
Compiling megakernel ...        # nvcc 编 ~2 分钟
Finished megakernel compilation...
<连贯的生成文本>
```

> ⚠️ 这里有个大坑:**`[MPK INIT]` 之后是 ~2 分钟的 nvcc 编译**,期间 GPU 也会占用。我最初以为它"卡死"了,提前 kill,误判成死锁——其实只是在编译。

巨核跑出了正确输出。但第一版(只能用 8 worker)是 68.2 ms/token,**比 native 慢**——因为只用了 64 个 SM 里的 8 个,且用的是 MPK 通用 ampere kernel,而非 PPU 优化库。

![megakernel 编译 + MPK INIT](/images/mpk-ppu/05-mpk-init-compile.png)

*配图 5：国产卡上真的生成并编译了 14140 任务的巨核 —— `[MPK INIT] Total tasks: 14140 ...` + 编译提示。*

---

## 六、死锁悬案:worker 一多就挂

想用更多 worker 提速时,诡异的事发生了:**8 worker 能跑,≥24 worker 必死锁**——PPU 100% 占用空转,最后报 `unspecified launch failure`。

![死锁现场 —— 一张卡 100% 空转](/images/mpk-ppu/06-deadlock-100util.png)

*配图 6：死锁现场 —— ppu-smi 里某一张卡 100% util、36GB 显存占着,却永远不出结果。这就是死锁的"罪案现场"。*

---

## 七、侦探工作:用 PPU 调试工具破案

这才是最有意思的部分。PPU 有一整套对标 NVIDIA 的调试工具,我逐一上阵:

**1. ppu-gdb attach 死锁进程**——发现所有 device warp 全是 worker,卡在 `execute_worker` 自旋等任务,**一个 scheduler 帧都没有**。

![ppu-gdb backtrace —— worker 全卡住、scheduler 不见踪影](/images/mpk-ppu/07-ppu-gdb-backtrace.png)

*配图 7：ppu-gdb backtrace —— `execute_worker` / `persistent_kernel.cuh` 占满、没有 `execute_scheduler`,worker 在空等一个没在跑的调度器。*

**2. racecheck**(对标 compute-sanitizer)——**0 hazard**,排除数据竞争。

**3. synccheck**(查 `__syncthreads`/barrier 死锁,跑了 13 分钟)——**0 报错**,排除 barrier 误用。

![racecheck / synccheck 干净通过](/images/mpk-ppu/08-racecheck-synccheck.png)

*配图 8：`RACECHECK SUMMARY: 0 hazard(s)` —— 不是内存竞争,synccheck 同样干净,排除 barrier 误用。*

**4. cooperative 占用探针**——意外发现 **PPU 其实是 256KB shared memory/SM**(MPK 按 sm_80 假设的是 163KB),且 64 个巨核 block 完全能同时驻留,排除"共驻留超限"。

![占用探针 —— 256KB/SM,64 block 全驻留](/images/mpk-ppu/09-coop-probe-256kb.png)

*配图 9：占用探针 —— `smemPerSM=256KB` + `N=8..64 launch=no error sync=no error`,推翻"放不下"的猜测。*

---

## 八、真相与修复:从"两个核"到"一个核"

破案的关键在 launch 代码里:MPK 默认 `split_worker_scheduler=true`,把 worker 和 scheduler 拆成**两个独立 kernel、两条 stream**,指望 PPU **并发执行两个常驻核**。

而 PPU 的并发多核调度比 N 卡保守:worker 数一多(≥~24),它就**不再把第二个 scheduler 核排上去** → scheduler 起不来 → worker 永久空等。这正是"只有 worker、没有 scheduler"的来源。

**修复一行:**

```cpp
// include/mirage/persistent_kernel/persistent_kernel.cuh:1617
- global_runtime_config.split_worker_scheduler = true;
+ global_runtime_config.split_worker_scheduler = false;
```

切到**单 kernel 路径**:worker 和 scheduler 合进同一个核,按 `blockIdx` 分角色,天然同核共驻留,不再依赖"并发两个核"。

结果:**单核 16 worker = 35.5 ms/token,反超 native 39.4 ms!**

![修复后 megakernel 反超 native](/images/mpk-ppu/10-fix-beats-native.png)

*配图 10：切到单核路径,16 worker 跑出 `a single persistent kernel` + 连贯生成 + `per-token latency: 35.x ms` —— 巨核在国产卡上反超自家优化栈的高光时刻。*

---

## 九、未竟之谜:32w 的"慢性死锁"

修复把上限从 8w 抬到 16w 并反超 native,但 **32w 仍会挂**——而且是另一种死法。全套工具排除下来:不是占用、不是竞争、不是 barrier 误用、不是单点自旋,而是 worker"拿到零星任务又等、整体不收敛"的 **distributed livelock**(分布式活锁)。这种高层进度死锁,标准 sanitizer 抓不到。

突破它需要在 MPK 调度器的规模化任务分发上做源码级改动——适合带着这套排除证据提一个上游 issue。**当前 16w(反超 native)是生产可用的上限。**

![性能对比柱状图](/images/mpk-ppu/11-benchmark.png)

*配图 11：性能对比柱状图 —— 三根柱子,单核 16w 最矮(最快)。*

---

## 结语

| 路径 | per-token latency |
|---|---|
| Native(torch + PPU 优化库) | 39.4 ms |
| Megakernel 两核 8w | 68.2 ms |
| **Megakernel 单核 16w(修复后)** | **35.5 ms ✅** |

一颗为 NVIDIA 手写的 CUDA 巨核,在国产 PPU 上不仅跑通、算对,还反超了厂商自家的推理栈——靠的是 PPU 扎实的 CUDA 兼容(连 `__CUDA_ARCH__=800`、cooperative launch、nvshmem 等价的 sailSHMEM 都齐),加上一行切换调度模式的修复。

剩下的 32w 活锁,是国产卡保守的并发调度 × MPK 规模化调度器之间的一道更深的题——留给下一篇。
