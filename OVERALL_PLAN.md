# OpenPTV2 Overall Roadmap: Desktop App to Cloud-Native Platform

## Vision
To establish OpenPTV2 as the standard, default desktop application for Particle Tracking Velocimetry, while laying the architectural foundation to seamlessly scale workloads as batch jobs on multi-core workstations, high-performance computing (HPC) clusters, and cloud infrastructure.

---

## Phase 1: Solidifying the Default Desktop App

*The goal of this phase is to make the software robust, easy to install, and user-friendly for a single user on a desktop computer.*

### 1.1 GUI Modernization & Decoupling
- **GUI Maintenance**: Maintain and stabilize the current Enthought TraitsUI/Chaco based GUI, ensuring it works seamlessly with the new pure Python algorithms.
- **Decouple GUI from Logic**: Ensure all core processing logic is strictly separated from UI components, using a Model-View-Controller (MVC) or similar pattern.
- **Modern Parameter Management**: Completely phase out legacy `.par` files and `exec()` calls. Standardize exclusively on `.yaml` for configuration via the centralized `ParameterManager`.

### 1.2 Engine Parity & Robust Distribution
- **Engine Parity**: Guarantee that all three branches (compiled C library, Numba-accelerated Python, and pure Python) produce the exact same results. This is critical for fallback behavior.
- **Precompiled Binary Wheels**: Before distributing standalone packages, ensure the core C library can be precompiled into cross-platform binary wheels and seamlessly linked to the Python code.
- **Numba Integration**: Ensure Numba installs robustly across platforms, enabling the fallback sequence: use compiled C library -> if unavailable, use Numba-accelerated Python -> in the worst case, fall back to pure Python.


### 1.3 Developer Experience
- **Consolidated Documentation**: Move the software from fragmented internal documentation to a unified static site (e.g., MkDocs) with clear user tutorials and developer guides.
- **Automated Testing**: Expand the test suite to ensure the modern GUI components are automatically tested alongside the core algorithms.

---

## Phase 2: Workstation and Batch Processing

*The goal of this phase is to enable users to run large-scale jobs on powerful local workstations without manual GUI interaction.*

### 2.1 Headless Execution & CLI
- **Headless Mode**: Ensure the core tracking engine (`openptv2`) operates entirely independent of the GUI.
- **Robust CLI**: Develop a comprehensive Command Line Interface (e.g., `openptv track config.yaml --headless`). This allows users to script and automate their workflows.

### 2.2 Local Parallelization
- **Multi-Core Processing**: Implement Python `multiprocessing` or `concurrent.futures` to parallelize independent pipeline stages (e.g., target detection, frame correction) across all available CPU cores.
- **Memory Optimization**: Refactor intermediate data storage (like `rt_is` and `ptv_is` files) to use more efficient binary formats (e.g., HDF5 or Parquet) to prevent disk I/O bottlenecks during high-speed batch processing.

---

## Phase 3: Cluster and HPC Readiness

*The goal of this phase is to scale the software across multiple machines in a research cluster environment.*

### 3.1 Containerization
- **Dockerization**: Provide official, lightweight Docker images containing the OpenPTV2 headless engine.
- **Singularity/Apptainer**: Ensure compatibility with HPC container runtimes commonly used in academic and national lab clusters.

### 3.2 Distributed Computing
- **Distributed Framework Integration**: Integrate with distributed computing frameworks like **Dask** or **Ray**. This will allow the processing of massive datasets by chunking frames across multiple cluster nodes.
- **Job Scheduler Compatibility**: Provide boilerplate submission scripts for common cluster schedulers (SLURM, PBS, LSF).

---

## Phase 4: Cloud-Native Implementation

*The ultimate goal: open-source tracking as a scalable web service.*

### 4.1 Cloud I/O Abstraction
- **Object Storage Support**: Refactor the I/O layer (currently reliant on local filesystem paths) to use abstraction libraries like `fsspec`. This allows the engine to stream images directly from and write results to AWS S3, Google Cloud Storage, or Azure Blob.

### 4.2 Web Interface & API
- **Web Dashboard**: Develop a modern web-based monitoring dashboard (e.g., using Streamlit, Dash, or FastAPI + React) to visualize real-time tracking progress, review 3D particle trajectories, and tweak parameters remotely.
- **Serverless/Microservices architecture**: Expose the pipeline stages as discrete microservices, allowing researchers to upload a batch of images and trigger a serverless tracking pipeline.
