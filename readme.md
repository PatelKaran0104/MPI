# Parallel Matrix Multiplication — MPI (SUMMA)

This project implements parallel matrix multiplication using the SUMMA
(Scalable Universal Matrix Multiply Algorithm) with MPI across multiple nodes.

## Project Structure

```
karan_Exam_2/
├── matmul.c                    # matrix multiplication using 1D row-wise data decomposition
├── matmul_v2.c                 # Alternate implementation
├── matrix.c                    # Primary SUMMA-based parallel matrix multiplication
├── slurmSubmit.sh              # SLURM script for speedup measurements
├── makefile                    # Build configuration
├── performance_analysis.tex    # LaTeX performance report
└── readme.md                   # This file
```

## Compilation

### Build

```bash
make
```

### Build a Specific Target

```bash
make matmul_v2
```

### Clean Build Artifacts

```bash
make clean
```

### Manual Compilation

```bash
mpicc -O3 -Wall -o matmul matmul.c -lm
mpicc -O3 -Wall -o matmul_v2 matmul_v2.c -lm
mpicc -O3 -Wall -o matrix matrix.c -lm
```

## Execution

**Usage:**
```bash
mpirun -np <num_processes> ./<binary_filename> <n> <seed> <verbose>
```

| Argument | Description |
|----------|-------------|
| `n` | Matrix size (n × n) |
| `seed` | Seed for the random number generator |
| `verbose` | Print matrices A, B, C if `1` and `n ≤ 10`; only checksum if `0` |

**Example — small matrix with output:**
```bash
mpirun -np 4 ./matrix 4 42 0
```
**Example — small matrix with verbose output:**
```bash
mpirun -np 4 ./matmul 4 42 1
```

**Expected output format:**
```
Matrix A:          ← only if verbose=1 and n≤10
...
Matrix B:          ← only if verbose=1 and n≤10
...
Matrix C (Result): ← only if verbose=1 and n≤10
...
Checksum: 127994455599.950012
Execution time with 512 ranks: 91.51 s
```

## SLURM Execution (HPC) For performance measurements, use the provided `slurmSubmit.sh` script. It is configured to run the executables with varying node counts and ranks.

```bash
sbatch slurmSubmit.sh
```

The script runs 5 repetitions for each node count (1, 2, 4, 6, 8 nodes × 64
ranks/node) and prints timing results for each run.

Slurm output: `matrix.out.<jobid>`