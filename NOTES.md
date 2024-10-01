# CUDA Migration Notes

---

## General Observations
- Summary of the original code's purpose and key functionality.
- Key differences between OpenMP and CUDA regarding parallelism and memory management.

---

## Complications Encountered

### Memory Management
- **Dynamic vs. Static Allocation**: 
  - Issues related to the allocation of device memory (`cudaMalloc` vs. `malloc`).
- **Unified Memory**: 
  - Considerations for using managed memory (`cudaMallocManaged`) vs. explicit device memory management.

### Kernel Design
- **Kernel Launch**:
  - Differences in how parallelism is expressed (thread/block configuration).
  - Limitations on thread and block sizes.
- **Data Dependencies**:
  - Challenges arising from data dependencies in loops and how they affect kernel design.
  
### Random Number Generation
- **CURAND Initialization**: 
  - How to properly initialize random number states within CUDA kernels.
- **Thread Safety**:
  - Issues with ensuring that random number generation is thread-safe.
