#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"
#include "_main.hxx"

using std::vector;
using std::swap;
using std::min;




// PAGERANK-FACTOR
// ---------------

template <class T>
__global__ void pagerankFactorKernel(T *a, const int *vdata, int i, int n, T p) {
  DEFINE(t, b, B, G);
  for (int v=i+B*b+t; v<i+n; v+=G*B) {
    int d = vdata[v];
    a[v] = d>0? p/d : 0;
  }
}

template <class T>
__host__ __device__ void pagerankFactorCu(T *a, const int *vdata, int i, int n, T p) {
  int B = BLOCK_DIM_M<T>();
  int G = min(ceilDiv(n, B), GRID_DIM_M<T>());
  pagerankFactorKernel<<<G, B>>>(a, vdata, i, n, p);
}




// PAGERANK-BLOCK
// --------------

template <class T, int S=BLOCK_LIMIT>
__global__ void pagerankBlockKernel(T *a, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  DEFINE(t, b, B, G);
  __shared__ T cache[S];
  for (int v=i+b; v<i+n; v+=G) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t==0) a[v] = c0 + cache[0];
  }
}

template <class T>
__host__ __device__ void pagerankBlockCu(T *a, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0, int G, int B) {
  pagerankBlockKernel<<<G, B>>>(a, c, vfrom, efrom, i, n, c0);
}


template <class G, class J>
auto pagerankWave(const G& xt, J&& ks) {
  vector<int> a {int(ks.size())};
  return a;
}




// PAGERANK (CUDA)
// ---------------

template <class T>
__global__ void pagerankCudaLoopKernel(int *i0, T *t0, T *a, T *r, T *c, T *f, const int *vfrom, const int *efrom, const int *vdata, int i, int n, int N, T p, T E, int L, int GP, int BP) {
  DEFINE(t, b, B, G);
  UNUSED(B); UNUSED(G);
  if (t>0 || b>0) return;
  int l = 1;
  volatile T *vt0 = t0;
  pagerankFactorCu(f, vdata, 0, N, p);
  for (; l<L; l++) {
    multiplyCu(c+i, r+i, f+i, n);
    sumIfNotInplaceCu(t0, r, vdata, N);
    cudaDeviceSynchronize();
    T c0 = (1-p)/N + p*(*t0)/N;
    pagerankBlockCu(a, c, vfrom, efrom, i, n, c0, GP, BP);
    l1NormInplaceCu(t0, r+i, a+i, n);
    if (*vt0 < E) break;
    swapCu(a, r);
  }
  *i0 = l;
}


template <class H, class T=float>
PagerankResult<T> pagerankCuda(H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o=PagerankOptions<T>()) {
  T   p = o.damping;
  T   E = o.tolerance;
  int L = o.maxIterations, l;
  int N = xt.order();
  int B = o.blockSize;
  int G = min(N, o.gridLimit);
  int R = reduceSizeCu<T>(N);
  auto ks    = vertices(xt);
  auto ns    = pagerankWave(xt, ks);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int N1 = N * sizeof(T);
  int R1 = R * sizeof(T);
  int I1 = 1 * sizeof(int);
  vector<T> a(N), r(N);

  T *t0D, *fD, *rD, *cD, *aD;
  int *i0D, *vfromD, *efromD, *vdataD;
  // TRY( cudaProfilerStart() );
  TRY( cudaMalloc(&i0D, I1) );
  TRY( cudaMalloc(&t0D, R1) );
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMalloc(&rD, N1) );
  TRY( cudaMalloc(&cD, N1) );
  TRY( cudaMalloc(&fD, N1) );
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  TRY( cudaMalloc(&vdataD, VDATA1) );
  TRY( cudaMemcpy(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice) );

  float t = measureDurationMarked([&](auto mark) {
    if (q) r = compressContainer(xt, *q, ks);
    else fill(r, T(1)/N);
    TRY( cudaMemcpy(aD, a.data(), N1, cudaMemcpyHostToDevice) );
    TRY( cudaMemcpy(rD, r.data(), N1, cudaMemcpyHostToDevice) );
    mark([&] { pagerankCudaLoopKernel<<<1, 1>>>(i0D, t0D, aD, rD, cD, fD, vfromD, efromD, vdataD, 0, N, N, p, E, L, G, B); });
    mark([&] { TRY( cudaDeviceSynchronize() ); });
  }, o.repeat);
  TRY( cudaMemcpy(&l,       i0D,        I1, cudaMemcpyDeviceToHost) );
  TRY( cudaMemcpy(a.data(), l&1? aD:rD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(i0D) );
  TRY( cudaFree(t0D) );
  TRY( cudaFree(aD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  TRY( cudaFree(vdataD) );
  // TRY( cudaProfilerStop() );
  return {decompressContainer(xt, a), l, t};
}
