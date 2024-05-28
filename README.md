# High-Performance Matrix Computations with StarPU
<div align="center">


</div>
Optimizing matrix multiplication using StarPU’s dynamic task scheduling for enhanced performance on heterogeneous architectures.
This project explores the use of StarPU, a task-based runtime system, to optimize matrix multiplication across CPUs and GPUs. By leveraging StarPU's capabilities, we aim to improve computational efficiency and resource utilization.

## Key Features
<li>Dynamic Task Scheduling: Efficiently manage tasks across CPUs and GPUs.
<li\>
<li>
Codelets: Reusable computation units for parallel execution.
<li\>
<li>Benchmarking: Performance analysis through extensive experiments.<li\>

## Further Reading

For a comprehensive understanding of the project's implementation, experiments, and results, please refer to the detailed [project report](docs/Report_CESISTA_HOUSSENBAY.pdf). This document provides in-depth insights and analysis that can help you fully grasp the scope and impact of the work.

## Acknowledgements
I would like to express my sincere gratitude to my university teacher, [ATTE Tori](https://fr.linkedin.com/in/atte-torri-37a08b133) for their guidance and support throughout the development of this project.

## Cloning the repository
1. Copy the git link from the repository page as shown below

<img src="https://github.com/TER-StarPU/ter-starpu-gemm/assets/14825656/444504cc-d069-4664-86d5-58f141dc2883" width="300"/>

2. In your terminal insert the following command

```bash
git clone --recurse-submodules [link]
```

## Some git basics
- Pulling updates from the remote

```bash
git pull
```

- Pushing updates to the remote

```bash
git add .
git commit -m "describe commit"
git push
```

- Creating a new branch

```bash
git branch [name]
git checkout [name]
```

- Merging branches can be done using the github web UI

## Updating the fork
To update your groups private fork, simply use the github web UI as shown below

<img src="https://github.com/TER-StarPU/ter-starpu-gemm/assets/14825656/cf7b93a9-5456-40d8-8167-662a315ada99" width="300"/>

## Building the project
The project is built with CMake

```bash
cmake -B build [-DOPTION=VALUE]
cd build
make -j
```

Where `OPTION` can be
- `ENABLE_CUDA` to build project with CUDA support (`VALUE` = `ON`/`OFF`, defaults to `OFF`)
- `ENABLE_MPI` to build project with MPI support (`VALUE` = `ON`/`OFF`, defaults to `OFF`)

## Execution
To execute the built program simply run
```bash
./gemm [args]
```

For help with program arguments
```bash
./gemm --help
```
