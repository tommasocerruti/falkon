# Kernels at large scale

## Kernel Methods

In supervised learning we observe examples $(x_i,y_i)_{i=1}^n$ and want to learn a function $f$ that predicts unseen data well. Kernel methods are the most popular non-parametric learning methods, that is not fixing the number of parameters of the model, letting us model highly non-linear relationships.

We embed each input through a (possibly infinite-dimensional) feature map

$$
\phi: X \;\longrightarrow\; \mathcal H,
$$

and run an ordinary linear algorithm in the feature space $\mathcal H$.

Here $\mathcal H$ is the reproducing-kernel Hilbert space (RKHS) induced by the kernel $K$.
It is a Hilbert space of real-valued functions on $X$ in which each *kernel section* $K(\cdot,x)$ lies in $\mathcal H$ and the reproducing property

$$
f(x)=\langle f,\,K(\cdot,x)\rangle_{\mathcal H}\qquad\forall f\in\mathcal H,\;x\in X
$$

holds. Because evaluating $f$ at a point is just an inner product with $K(\cdot,x)$, any solution of an empirical-risk problem can be expressed as a finite linear combination of these sections:

$$
f(\cdot)=\sum_{i=1}^{n}\alpha_i\,K(x_i,\cdot).
$$

Hence we only have to learn the coefficients $\alpha$; the explicit feature vectors $\phi(x)$ never appear.  More generally, every symmetric positive-definite kernel satisfies

$$
K(x,x')=\langle\phi(x),\phi(x')\rangle_{\mathcal H},
$$

so we can substitute the scalar value $K(x,x')$ wherever a feature-space inner product would occur.
This substitution is the kernel trick (e.g. with the Gaussian/RBF kernel).

In pratice we:

1. Choose a positive-definite kernel (say, Gaussian/RBF) and build the Gram matrix
   $K\in\mathbb R^{n\times n}$ with entries $K_{ij}=K(x_i,x_j)$.

2. Fit a regularised linear model in feature space.
   For kernel ridge regression (squared loss + $\ell_2$ regularisation) we solve

   $$
   \min_{\alpha\in\mathbb R^{n}}
     \frac1n\bigl\|K\alpha-y\bigr\|_2^2
     \;+\;
     \lambda\,\alpha^{\!\top}K\alpha.
   $$

   By the Representer Theorem, every minimiser has the finite-sample form
   $f(\cdot)=\sum_{i=1}^{n}\alpha_i K(x_i,\cdot)$.

3. Solve the linear system

   $$
   (K+n\lambda I)\,\hat\alpha \;=\; y,
   $$

   yielding the coefficients $\hat\alpha$.

4. Predict a new point $x*$ via
   $$
   f(x*) \;=\; \sum_{i=1}^{n}\hat\alpha_i\,K(x_i,x*).
   $$

However, this pipeline becomes prohibitive at two points:

* Constructing the Gram matrix $K$ requires computing and storing every entry $K(x_i,x_j)$, which costs $O(n^{2})$ time and $O(n^{2})$ memory.
* Solving the linear system $(K+n\lambda I)\alpha = y$ via Cholesky factorisation on an $n\times n$ matrix costs $O(n^{3})$ time.

While such costs are acceptable for datasets with only a few thousand examples, they explode when $n$ reaches hundreds of thousands or millions. To overcome these limits, scalable kernel methods, including FALKON, EigenPro and related approaches, compress or precondition the problem so that kernel learning remains practical on modern, large-scale data.

---
---

## FALKON [1 - Basic Algorithm](https://arxiv.org/pdf/1705.10958)

FALKON (Fast Approximate Large-scale Kernel Ridge Regression) keeps the statistical strength of exact KRR while pushing its cost down to almost-linear in $n$.
It does so through three successive steps.

### 1 – Nyström approximation

Select $m\!\ll\!n$ centers $Z=\{z_j\}_{j=1}^{m}$ (uniformly or via leverage scores) and form the two kernel blocks

$$
K_{nm}=K(X,Z), \qquad K_{mm}=K(Z,Z).
$$

Replace the full Gram matrix by its rank-$m$ Nyström proxy

$$
\tilde K \;=\; K_{nm}\,K_{mm}^{-1}\,K_{nm}^{\!\top}.
$$

This means that only $K_{nm}$ (size $n\times m$) and $K_{mm}$ (size $m\times m$) are stored:
memory and matrix–vector products drop from $O(n^{2})$ to $O(nm)$, where m is typically choosen as $m = \sqrt n$.

### 2 – Pre-conditioning

Poor conditioning slows any iterative solver.
FALKON builds a left pre-conditioner

$$
B \;=\; \frac{1}{\sqrt n}\,K_{mm}^{-1/2}\,K_{nm}^{\!\top},
$$

and rewrites the KRR system

$$
(K+n\lambda I)\alpha = y
$$

as

$$
\bigl(BB^{\!\top} + \lambda I\bigr)\,\alpha \;=\; B\,y.
$$,

where the condition number of $
\bigl(BB^{\!\top} + \lambda I\bigr) \alpha
$ is $O(1)$.

This allows Conjugate Gradient to reach a high-precision solution in a constant number of iterations (typically 5–15) instead of thousands; each step inherits the $O(nm)$ cost from Nyström.

### 3 – Conjugate-gradient iteration

Conjugate Gradient (CG) is an iterative solver for symmetric positive-definite systems that needs only

* one matrix–vector product with $BB^{\!\top}$
  $(v\mapsto K_{nm}(K_{mm}^{-1}(K_{nm}^{\!\top}v)))$ — $O(nm)$;
* a few inner products and vector additions — $O(n)$.

No dense $n\times n$ matrix is ever formed or factorised, which is the real bottleneck of KRR.

### Putting it all together

* one-off costs: build $K_{nm}$ and $K_{mm}$ [$O(nm)$ time / memory] and factorise $K_{mm}+m\lambda I$ [$O(m^{3})$ time].
* iterative solve: just few steps of CG thanks to the pre-conditioner [$O(nm)$ time per CG step].
* final predictor: stores just the $m$ landmarks and their weights

$$
f(x*) \;=\; \sum_{j=1}^{m}\beta_j\,K(z_j,x*).
$$

With $n=1M$ and $m=5k$, the full KRR problem that would have needed $8 TB$ and days to solve, now fits in ≈ $40 GB$ and trains in minutes, yet retaining near to optimal accuracy. 

---
---

## FALKON [2 - Multi-GPU parallelism](https://arxiv.org/pdf/2006.10350)

Modern GPUs deliver trillions of floating-point operations per second, yet offer only a few GB of memory and low bandwidth. A vanilla FALKON implementation is therefore bandwidth-bound rather than compute-bound: it streams huge kernel blocks back and forth while performing only a handful of flops per byte (operational intensity is low). We need to reconstruct FALKON so that it:  
(i) fits into workstation-class GPUs  
(ii) keeps those GPUs busy  
(iii) scales almost linearly with the number of accelerators  
The redesign revolves around five tricks.

---

### 2.1 One-buffer preconditioner (RAM bottleneck removed)

Using Nystrom method we need to store the matrices $K_{mm}$ and $K_{nm}$. However $K_{nm}$ is only used in matrix-vector products, therefore we can avoid to construct it explicitly, so the only dense object that must ever reside in CPU RAM is $K_{mm}$.
FALKON now allocates a single matrix of this size and overwrites it in place while building the Cholesky-based preconditioner

$$
P \;=\;\frac1{\sqrt n}\,T^{-1}A^{-1},\qquad 
T=\operatorname{chol}(K_{mm}),\;
A=\operatorname{chol}\!\Bigl(\tfrac1mTT^{\!\top}+\lambda I\Bigr).
$$

Because $K_{mm}=T^{\!\top}T$, all subsequent occurrences of $K_{mm}$ in the linear system cancel, so the matrix is never needed again.  Roughly 90% of the total RAM footprint is now the preconditioner itself; everything else (two work buffers and the weight vector) is negligible.

---

### 2.2 Streaming $K_{nm}$ on the GPU (no full Gram block)

Matrix-vector products of the form $v\mapsto K_{nm}(K_{mm}^{-1}(K_{nm}^{\!\top}v))$ drive every CG step.
Rather than materialising $K_{nm}$ (which would take $n m$ memory), the input matrix $X\in\mathbb R^{n\times d}$ is sliced into batches of $q$ rows:

$$
\sum_{b=1}^{B} k(X_{b,:},Z)^\top\;\bigl(k(X_{b,:},Z)\,v\bigr).
$$

Each batch

1. is copied to a free GPU,
2. has its kernel block computed on device,
3. multiplies the block by the current vector,
4. returns only a length-$m$ result to the host.

Thus the full $K_{nm}$ never exists simultaneously either in host or device memory.  The batch sizes $(q,r,s)$ along $(n,m,d)$ are tuned to maximise compute/transfer ratio while respecting the GPU’s memory budget.

---

### 2.3 Out-of-core, multi-GPU Cholesky

The landmark block may still exceed a single GPU’s memory when $m$ is large.  FALKON factorises it tile-by-tile across all available GPUs in a 1-D block-cyclic pattern:

1. one device factorises the diagonal tile $A_{ii}$;
2. all devices solve triangular systems for their assigned off-diagonal tiles $A_{ji}$;
3. each device updates its slice of the trailing matrix $A_{jj'}$.

Only one tile column plus one extra tile sits on any GPU at a time, so the algorithm scales to virtually arbitrary $m$ while achieving high utilisation of multiple accelerators.

---

### 2.4 Hiding data transfers

Each GPU has three independent engines: *copy-in*, *compute*, *copy-out*.  FALKON runs the three activities in parallel streams; while a batch is being processed, the next batch is already loading and the previous result is already returning (see Fig. 3 in the paper).  In the ideal balanced case $t$ batches finish in $t+2$ latency units instead of $3t$.

---

### 2.5 Precision tricks and edge cases

* **Mixed precision** Kernel blocks are generated in FP32 for speed, temporarily cast to FP64 only when summing large norms to avoid catastrophic cancellation, and cast back to FP32 for storage—regaining a ≈ 10× throughput boost without losing positive-definiteness.
* **Thin blocks** When a batch becomes “skinny”, cuBLAS cannot saturate the GPU.  FALKON falls back to KeOps, which is optimised for tall-skinny kernel products.
* **Sparse inputs** High-dimensional text data remain in CSR/COO format; GPU-side sparse kernels and norms are provided so that no dense conversion is needed.

---

### 2.6 End-to-end complexity and speed-ups

* **Memory** $m(m+2d)+qd$ floats on the host; at most one tile column plus a batch on each GPU.
* **Compute** Initial factorisation $O(m^{3})$; each CG step $O(nm)$ but only 5–15 steps thanks to conditioning $O(1)$.
* **Throughput** On a workstation with four × 16 GB GPUs, $n=10^7,\;m=2\times10^5$ fits comfortably and trains in a few hours, whereas the unmodified algorithm would require terabytes of RAM and be hopelessly slow.

With these engineering changes, FALKON preserves its original statistical guarantees yet fully exploits today’s multi-GPU hardware, achieving state-of-the-art accuracy on datasets that were previously considered out of reach for exact kernel methods.