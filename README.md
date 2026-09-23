## CUDA H2D/D2H bandwidth benchmark: pageable `malloc` vs. pinned `cudaHostAlloc`, 1 MB–1 GB sweep

This project measures the effective bandwidth of memory transfers between host (CPU) memory and device (GPU) memory. It sweeps transfer sizes from 1 MB to 1 GB in powers of two and compares, in both transfer directions, two ways of allocating the host buffer:

| Host memory type | Allocation API | Description |
|---|---|---|
| Pageable | `malloc` | Standard virtual memory that the OS may page out |
| Pinned (page-locked) | `cudaHostAlloc` | Memory locked in physical RAM, directly accessible by the GPU's DMA engines |

Four configurations are measured at each size: H2D pageable, D2H pageable, H2D pinned, and D2H pinned. Results are written as CSV and plotted as bandwidth vs. transfer size.

---

## Key Results

From the sample run in this repository, at large transfer sizes (≥ 128 MB):

| Configuration | Bandwidth (GB/s) | Behavior |
|---|---|---|
| H2D pinned | 16.9 – 17.4 | Stable plateau; highest sustained throughput |
| D2H pinned | 15.0 – 15.4 | Stable plateau |
| D2H pageable | 13.3 – 13.4 | Stable, ~10–13% below pinned |
| H2D pageable | 4.5 – 13.7 | Highly variable; as low as ~26% of pinned throughput |

Pinned memory delivers the highest and most consistent bandwidth for large transfers. The effect is strongest for host-to-device copies: at 256 MB, pinned H2D reaches 17.0 GB/s against 4.5 GB/s for pageable, a 3.8× difference.

---

## Background: Why Host Memory Type Matters

The GPU copies data using DMA, which requires the host buffer to stay at a fixed physical address for the duration of the transfer. Pageable memory offers no such guarantee, so for a pageable transfer the CUDA driver first copies the data into an internal pinned staging buffer and then performs the DMA from there. This extra host-side copy consumes CPU memory bandwidth and adds overhead.

With pinned memory allocated through `cudaHostAlloc`, the DMA engine reads from or writes to the user's buffer directly, eliminating the staging copy. The trade-off is that pinned memory is a limited system resource: allocating too much of it reduces the physical memory available to the OS and other processes.

---

## Methodology

For each transfer size and configuration, the benchmark allocates a host buffer (pageable or pinned) and a device buffer of equal size, performs the transfer in the specified direction, and computes effective bandwidth as bytes transferred divided by elapsed transfer time.

| Parameter | Values |
|---|---|
| Transfer sizes | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 MB (11 sizes, powers of two) |
| Directions | Host-to-device (H2D), device-to-host (D2H) |
| Host memory | Pageable (`malloc`), pinned (`cudaHostAlloc`) |
| Metric | Effective bandwidth in GB/s |

All CUDA API calls are wrapped in the `CHECK_CUDA` error-checking macro defined in `cuda_utils.cuh`.

---

## Project Structure

```text
.
├── main.cu              # Entry point: runs all configurations and prints CSV to stdout
├── bandwidth.cu         # Bandwidth measurement implementation
├── bandwidth.h          # Measurement API and data structures
├── cuda_utils.cuh       # CHECK_CUDA error-checking macro
├── Makefile             # nvcc build script
├── plot_bandwidth.py    # Generates the bandwidth vs. transfer size plot
├── report.tex           # LaTeX write-up (optional)
├── bandwidth.csv        # Generated: raw measurement data
└── bandwidth_plot.png   # Generated: bandwidth vs. transfer size figure
```

---

## Requirements

**Hardware**

- NVIDIA GPU with CUDA support
- At least 2 GB of free host memory and 1 GB of free device memory for the largest transfer size

**Software**

- CUDA Toolkit, including `nvcc`
- A host C++ compiler supported by your CUDA version
- Python 3 with `numpy` and `matplotlib` (for plotting)
- A LaTeX distribution (optional, for compiling `report.tex`)

Install the Python dependencies with:

```bash
pip install numpy matplotlib
```

---

## Build

From the project root:

```bash
make
```

This compiles `main.cu` and `bandwidth.cu` with `nvcc` and produces the `bandwidth` executable. To remove build artifacts:

```bash
make clean
```

---

## Usage

### 1. Run the benchmark

```bash
./bandwidth > bandwidth.csv
```

The program runs all 44 measurements (11 sizes × 2 directions × 2 memory types) and writes CSV to standard output with the following header:

```text
size_MB,h2d_pageable_GBs,d2h_pageable_GBs,h2d_pinned_GBs,d2h_pinned_GBs
```

Inspect the output with:

```bash
head bandwidth.csv
```

### 2. Generate the plot

```bash
python3 plot_bandwidth.py
```

This reads `bandwidth.csv` and writes `bandwidth_plot.png`, containing four curves: H2D pageable, D2H pageable, H2D pinned, and D2H pinned.

### 3. Compile the report (optional)

```bash
pdflatex report.tex
```

---

## Results

Effective bandwidth (GB/s) from a sample run:

| Size (MB) | H2D Pageable | D2H Pageable | H2D Pinned | D2H Pinned |
|---:|---:|---:|---:|---:|
| 1 | 3.846 | 2.118 | 2.506 | 4.864 |
| 2 | 11.652 | 6.564 | 9.892 | 8.520 |
| 4 | 13.352 | 7.090 | 10.056 | 8.928 |
| 8 | 19.567 | 12.200 | 22.272 | 15.912 |
| 16 | 20.608 | 12.727 | 23.510 | 15.352 |
| 32 | 19.644 | 13.076 | 12.188 | 15.112 |
| 64 | 8.120 | 9.064 | 15.961 | 15.407 |
| 128 | 5.152 | 13.316 | 16.913 | 15.363 |
| 256 | 4.525 | 13.428 | 17.027 | 15.364 |
| 512 | 4.494 | 13.397 | 17.120 | 15.197 |
| 1024 | 13.679 | 13.396 | 17.392 | 15.021 |

![Host-device transfer bandwidth vs. transfer size](bandwidth_plot.png)

---

## Analysis

**Small transfers (1–4 MB).** Bandwidth is low and inconsistent across all configurations. At these sizes, fixed per-transfer costs and timing variance make up a large share of each measurement, and the ordering between configurations is not reliable: pageable H2D exceeds pinned H2D at every size in this range.

**Medium transfers (8–32 MB).** Bandwidth rises sharply. The highest individual readings in the sweep occur here (23.5 GB/s pinned H2D and 20.6 GB/s pageable H2D at 16 MB), exceeding the large-transfer plateau. Because these peaks are neither sustained nor monotonic (pinned H2D drops to 12.2 GB/s at 32 MB), they are best treated as measurement variance rather than true hardware throughput.

**Large transfers (≥ 64 MB).** The pinned configurations settle into stable plateaus of about 17 GB/s for H2D and 15 GB/s for D2H, varying by less than 1 GB/s from 128 MB to 1 GB. Pageable D2H is similarly stable at about 13.4 GB/s. Pageable H2D is the outlier: it falls to about 4.5 GB/s between 128 and 512 MB before recovering to 13.7 GB/s at 1 GB, consistent with the extra staging copy making pageable transfers sensitive to host-side memory behavior.

**Direction asymmetry.** With pinned memory, H2D is consistently about 2 GB/s faster than D2H at large sizes. Asymmetries of this kind are common and depend on the platform's chipset, interconnect, and driver.

---

## Practical Takeaways

- Use pinned memory (`cudaHostAlloc` or `cudaMallocHost`) for large, performance-critical transfers, especially host-to-device.
- Pinned memory is also a prerequisite for overlapping transfers with kernel execution using `cudaMemcpyAsync` and CUDA streams.
- Batch many small transfers into fewer large ones; small transfers do not reach the achievable bandwidth.
- Allocate pinned memory selectively, since excessive page-locking reduces memory available to the rest of the system.

---

## Limitations

- **Single sample run.** The results above come from one run of the benchmark. Several readings (notably the medium-size peaks and the pageable H2D dip) vary more than expected, so repeating each measurement and reporting the median would give more reliable figures.
- **Platform dependence.** Absolute bandwidth depends on the GPU, the host–device interconnect, the CPU and memory subsystem, and the driver version. The trends between configurations are more transferable than the specific numbers.
