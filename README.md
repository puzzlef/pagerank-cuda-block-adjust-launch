Comparing various launch configs for **CUDA block-per-vertex** based PageRank
([pull], [CSR]).

This branch uses 2-stream based pagerank-loop kernel which runs on GPU instead.
Results indicate this is **slower** than simple pagerank-loop on the CPU
([static]), and **equal speed** as simple pagerank-loop on GPU ([dynamic]).

All outputs are saved in [out](out/). The input data used for this experiment
is available at ["graphs"] (for small ones), and the [SuiteSparse Matrix Collection].
This experiment was done with guidance from [Prof. Dip Sankar Banerjee] and
[Prof. Kishore Kothapalli].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/min-1DeadEnd.mtx
$ ./a.out ~/data/min-2SCC.mtx
$ ...

# ...
```

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://www.slideshare.net/SubhajitSahu/cuda-by-example-notes)
- [compilation .cu files with Dynamic Parallelism(CUDA)](https://stackoverflow.com/a/27870808/1413259)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [nvGraph pagerank example, EN605.617, JHU-EP-Intro2GPU](https://github.com/JHU-EP-Intro2GPU/EN605.617/blob/master/module9/nvgraph_examples/nvgraph_Pagerank.cpp)
- [nvGraph pagerank example, CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/10.0/nvgraph/index.html#nvgraph-pagerank-example)
- [nvGraph Library User's Guide](https://docs.nvidia.com/cuda/archive/10.1/pdf/nvGRAPH_Library.pdf)
- [RAPIDS nvGraph NVIDIA graph library][nvGraph]
- [SuiteSparse Matrix Collection]

<br>
<br>

[![](https://i.imgur.com/QIUy2ds.jpg)](https://www.youtube.com/watch?v=4EG2up-jcKM&t=12897s)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://suitesparse-collection-website.herokuapp.com
["graphs"]: https://github.com/puzzlef/graphs
[nvGraph]: https://github.com/rapidsai/nvgraph
[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[csr]: https://github.com/puzzlef/pagerank-class-vs-csr
[static]: https://github.com/puzzlef/pagerank-cuda-block-adjust-launch/tree/static
[dynamic]: https://github.com/puzzlef/pagerank-cuda-block-adjust-launch/tree/dynamic
