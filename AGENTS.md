# AGENTS.md

Project context for AI coding assistants working on molina. Tool-specific
workflow instructions live in each tool's own file, e.g. CLAUDE.md.

## What this is

molina is linear algebra in pure Mojo: `RealArray` vectors and dense
matrices, `CSRMatrix`, dense factorizations, sparse direct solvers (ports
of faer), Krylov solvers, shift-invert Lanczos, and smoothed-aggregation
multigrid (amgcl's aggregation). It was split from MojoDEC
(`../MojoDEC`), whose history it keeps; MojoDEC, mogeometry
(`../mogeometry`) and minigrad (`../minigrad`) depend on it as a
pixi-build package (see README.md).

## Commands

```sh
pixi run test                   # tests/test_*.mojo
pixi run test-release           # the same with -D ASSERT=none
pixi run package                # build/molina.mojoc
pixi run doc-check [--list PAT] # doc string warnings: must be none
pixi run docs                   # API reference in docs/api/
pixi run format                 # mojo format molina tests
mojo run -I . tests/test_build.mojo   # one test file (inside `pixi shell`)
```

The workspace is also a pixi-build package (`[package]` in `pixi.toml`).
Building it as a consumer's dependency needs pixi 0.84 or newer (0.70
fails on the pixi-build-mojo backend); its own tasks run on older pixi.

## Tech stack and toolchain

- `pixi.toml` asks for `mojo >=1.3.0.dev0,<2` (plain `>=1.3` would exclude
  the 1.3 nightlies, which conda orders before 1.3.0); `pixi.lock` records
  the build actually tested. Don't pin exact versions in `pixi.toml`. The
  consumers each compile molina with their own Mojo (their
  `[workspace.build-variants] mojo-compiler`), which may be older or newer
  than this lock's, so molina must not use features of the newest
  nightly only.
- molina imports nothing but Mojo's standard library: no MAX, no NuMojo,
  nothing from its consumers. minigrad runs it without MAX. Keep it so.
- `-D ASSERT=none` removes `debug_assert` bounds checks. `mojo run`/`build`
  accept it; `mojo precompile` does not, so it applies only to code
  compiled from source.

### Known toolchain traps

- `assert_almost_equal` has a default `rtol=1e-5` on top of `atol`, so
  every call passes `rtol` explicitly (normally `rtol=0`, with an `atol`
  from the problem's conditioning); `tests/run_all.sh` fails on a call
  without one.
- `RealArray`'s `+`, `-`, and scaling allocate their result: in hot loops,
  write the loop over `unsafe_get`/`unsafe_set` (the AMG cycle went from
  3 ms to 0.09).
- Write `sqrt(x)` (`std.math`), never `x ** 0.5`: Mojo's `**` with a
  fractional exponent is off by up to ~5e-12 relative, which Cholesky and
  Gram-matrix solves amplify.
- Mojo's float operations allow contraction, so LLVM may fuse `a * b + c`
  into one rounding, or not, depending on how the code around it is laid
  out. Where two code paths must agree bit for bit (the supernodal solve's
  tile and tail code), write the multiply-adds as explicit `x.fma(y, acc)`.
- Mojo's `>>` by 64 or more is undefined (it gave garbage): bound shift
  loops.

## Architecture

The modules, lowest first: `real_array` (`RealArray`: C-contiguous
`Float64`, rank 1 or 2, NuMojo's method names for flat access,
`+`/`-`/scaling entrywise, `to_numpy`) and `utils` (its builders and
level-1 kernels), `parallel`, `index_set` (the `IndexSet` trait:
`universe_size()` and `in`, which `CSRMatrix` and `DiagonalOperator`'s
`exclude_subset`/`restrict_to_subset` take; MojoDEC's `Subset` implements
it), `operator`, `diagonal`, `sparse`, `dense`, `ordering`, `supernodal`,
`sparse_ldlt`, `linalg`, `lanczos`, and `amg`. `molina/__init__.mojo`
re-exports the public names.

molina knows nothing of its consumers. MojoDEC re-exports these modules
from its own `routines/linalg.mojo`, `lanczos.mojo` and `sparse_ldlt.mojo`
and defines typed overloads of `cg`, `minres`, `jacobi`, `lsqr`,
`spsolve`, `eigh_generalized` and `eigsh` there, as forwarders to these
functions: changing a public signature here breaks those forwarders, so
change them together (and say so in the commit message).

- **Operators and Krylov solvers** (`molina/operator.mojo`,
  `molina/linalg.mojo`): `LinearOperator` is `num_rows`/`num_cols`/
  `apply_into(x, mut y)` on `RealArray`; `CSRMatrix`, `DiagonalOperator`,
  and `Identity` implement it (and MojoDEC's typed `MatrixOp`,
  `DiagonalOp`, and `BlockOp`). `Preconditioner` is a method-less marker for
  symmetric positive definite operators (`DiagonalOperator`, `Identity`,
  `SparseCholesky`; never a general `CSRMatrix`). `BlockDiagonal(P1, P2)`
  (`molina/operator.mojo`) stacks two for 2 x 2 systems; `DenseCholesky`
  (`molina/linalg.mojo`) is an exact dense inverse for small problems. `cg` and `minres` are generic
  over both, with `M=` keyword-only; the unpreconditioned overloads pass
  `Identity`, which reproduces the old iterates bit for bit. Both raise on
  a non-symmetric operator (`_require_symmetric`, two probe products) and
  stop with `BREAKDOWN` on `r^T M r < 0`. Keyword options on both:
  `x0=` (tolerance stays relative to `||b||`), `nullspace=` (a
  `NullSpace`, orthonormalized by Gram–Schmidt twice and shared behind an
  `ArcPointer`; removed from `b`, `x0`, and every preconditioned residual
  or Lanczos vector, and checked against `||A q|| <= sqrt(eps) ||A||`
  with `||A||` from the symmetry probes), and `ritz=True` (the Ritz
  values of the Lanczos tridiagonal, from CG's `alpha`/`beta` or MINRES's
  `alfa`/`beta`, by `dense.tridiagonal_eigenvalues`, QL without vectors).
  `eigenvalue_range` runs 10 CG steps on `probe_vector(n, 3)` for the
  extreme Ritz values (inner bounds). Without the options, every solve is
  bit for bit what it was before them; keep it so.
  `eigsh(K, M, nev, sigma=)` (`molina/lanczos.mojo`, from the papers:
  Ericsson–Ruhe, Wu–Simon) factors `K - sigma M` once (`SparseLDLT`)
  and runs thick-restart Lanczos on `OP = (K - sigma M)^-1 M` in the `M`
  inner product (basis and `M V` stored, full reorthogonalization twice,
  `H` dense, Ritz pairs by `dense.eigh`, accepted at `tol |theta|`).
  `_run` finds the `want` pairs of largest `|theta|` in the complement
  of a locked set; `eigsh` runs it once for `nev`, then repeatedly with
  the found pairs locked (`_PROBE = 4`) and swaps in anything nearer
  sigma (`_merge`), because single-vector Lanczos misses copies of
  multiple eigenvalues (the icosphere did). `deflate=G` projects every
  vector `M`-orthogonally off `range(G)` with `SparseCholesky(G^T M G)`.
  Eigenvectors come back `n x nev`; `residuals` and `mode` take `n x m`.
  `EighResult.residuals(K, M)` gives each
  pair's normwise backward error with norms from 30 power steps (lower
  bounds, so the errors err high); dense arrays are wrapped in a private
  `_Dense` operator. Pass sparse `K` to keep it at n sparse products.
- **Sparse direct** (`molina/ordering.mojo`, `molina/sparse_ldlt.mojo`
  `molina/supernodal.mojo`; ports of faer's AMD and simplicial and
  supernodal Cholesky, notices in `NOTICE`):
  `amd_ordering(A)` builds AMD's quotient graph exactly as faer's
  `aat`/`amd_1` do from the sorted pattern of `A + A^T`, so its
  permutations equal faer's (`tests/data/amd/`, generated by the Rust
  driver there); rows with an absent or zero diagonal are ordered last
  (treated as AMD's dense rows), which makes saddle points factor
  without pivoting. `SparseLDLT` checks symmetry entrywise (`1e-12` of
  the largest entry), takes the lower triangle of `P A P^T`, computes the
  elimination tree and column counts, then each row of `L` by `_ereach`
  (stack order is a valid topological order); `D[j]` is stored in place
  of `L`'s unit diagonal. Dynamic regularization (`signs`,
  `regularization`) as faer's. Every `solve` takes `refinement_steps`
  (default 1) of iterative refinement against a stored copy of `A`: the
  unpivoted indefinite factor is not backward stable, and MojoDEC's DarcyFlow
  pressures need the step to agree to 1e-14. `perm=` replaces the
  ordering; `cholesky_nonzeros(A, perm)` runs only the symbolic phase
  (`_PermutedLower`, `_etree_and_counts`, shared with the factor).
  `SparseCholesky` wraps it
  with no refinement, requires positive pivots, and is a
  `Preconditioner`. Simplicial factors and all solves are serial.
  `spsolve(..., method=DirectMethod.SPARSE_LDLT)`
  uses it.
  `layout=FactorLayout.AUTO` goes supernodal above 20 flops per entry
  of `L` (`_SUPERNODAL_RATIO`, measured break-even; faer uses 40): the
  permutation is relabeled by `etree_postorder` (same fill), and
  `_Supernodal` finds fundamental supernodes, merges each with the
  supernode before it in the postorder under faer's relaxation
  thresholds, finds patterns as faer's `ereach_super`, and factors
  left-looking: per supernode, scatter `C`, subtract each descendant's
  `L_b D L_m^T` (`_Frame.update`: a dense product into scratch,
  scattered by `rmap`), then the dense block by panels of 32 columns
  (`panel_update`: the columns before the panel in one product;
  `factor_diagonal`: the panel's diagonal block, left-looking, with the
  pivot rule; `panel_eliminate`: the rows below it). `_sub_product` is
  the one kernel: register tiles of 8 rows x 4 columns, lower trapezoid
  (`top` offsets the diagonal for blocks of rows below it). The numeric
  phase is a task list (`_Tasks`) run in one `parallel_ranges` call on
  the physical cores, above `_PARALLEL_WORK` (5e6 multiply-adds; serial
  below): Geist-Ng subtrees (heaviest first), then each supernode above
  them as one task, or, past `_SPLIT_WORK`, column windows (each applies
  every descendant's update to its columns), then per panel a diagonal
  task and `_ROW_BLOCK` (64) row tasks; a panel's diagonal task waits
  only for the row block holding its rows (and the panel two back,
  whose coefficient buffer it reuses). Threads claim tasks from an
  atomic counter in list order and spin on each one's pending count
  (`Atomic[Int64]` statics on untracked pointers); predecessors come
  first, so a waiting task's predecessors are held by running threads
  and it cannot deadlock. A call per panel instead cost 225 us each, 50
  ms at 30^3. In 3D under AMD the subtrees hold ~3% of the work: the
  split supernodes are what scale. Bits equal serial at every thread
  count (row and column splits keep each entry's operations and order);
  check a change with `test_parallel_supernodal_matches_serial`. A failed
  pivot sets a flag every thread polls, then the whole factorization
  reruns serially so the error is serial's (first in supernode order).
  `vals` is allocated uninitialized and each task zeroes the block or
  columns it assembles (parallel first touch). The frame's pointers are
  untracked: keep the lists they point into alive (`_ = x^`) past the
  parallel call. Solves gather each supernode's rows once. `solve(B)`
  (n x m) is `_solve_many`: `_solve_block(c_lo, c_hi)` per block of
  columns does everything for them: gathers `B` by columns (permuted),
  `_solve_columns(Y, c_lo, c_hi)` (each supernode, or simplicial column,
  applied to every column in turn; `_solve_factored` is its one-column
  case), scatters, and per refinement step the residuals (the loop of
  `CSRMatrix.matvec_into`, same sums), the solve, and `x += dx`; so
  column c is the single solve bit for bit. The supernodal
  `solve_columns` gathers each supernode's rows below its triangle for
  all the block's columns (`_gather`, deg x mb by columns) and multiplies
  as matrices: `_sub_product` (the factor's kernel, `W` = the
  supernode's solved rows, row-major) going down, `_sub_dots` (2 x 2
  tiles of `L` columns by `Y` columns, `_W` lanes then `reduce_add`,
  odd ones by `_dot_lanes`) going up. The triangles take groups of 4
  columns of `Y`: down by blocks of `_W` rows, the columns of `L`
  before a block by `_down_tiles` (8 x 4 register tiles starting from
  `Y`, a zero `x_j` skipped per column as the single solve skips it),
  then the block's own triangle; up by `_up_triangle4` (one load of `L`
  for four dots); the other `mb % 4` columns column by column (a
  one-column tile reads `L` across its rows, slower than the sweep down
  each column). Every entry's chain is fixed whatever `mb` or tile, so keep
  explicit `fma`s there (contraction differs between tile and tail
  code). `mb == 1` (single solves, and a core's one-column block) has
  its own loops with the same chains, built column by column of `L`
  (`_down_triangle`, `_up_triangle`, `_dot_lanes`) without `W` or
  `_sub_product`: through the matrix kernels a single solve was 20-28%
  slower. Above `_PARALLEL_SOLVE_WORK`
  ((stored + steps (stored + nnz A)) x m = 4e6) the blocks go to the
  physical cores in one `parallel_ranges` call. 30^3, m = 64, no
  refinement (i7-10875H): 915 ms as single solves, 304 ms one thread,
  53 ms on 8.
  3-7x faster factors than simplicial in 3D on one thread, about 4x more
  on 8 cores; results differ from simplicial in the last bits. The
  supernodal path relabels `etree` and `counts` by the postorder instead
  of recomputing them (an equivalent ordering). `_etree_and_counts` is
  Liu's elimination tree (ancestors path-compressed, in `visited`) and
  Gilbert-Ng-Peyton column counts (row-subtree leaves in a postorder,
  least common ancestors by a disjoint-set forest; bookkeeping as faer's
  `column_counts_aat`), nearly linear in nnz(C): 2.2 ms at 30^3 where
  walking every path of L took 13; `test_column_counts_match_dense_elimination`
  checks it against dense symbolic elimination.
- **Algebraic multigrid** (`molina/amg.mojo`): `SmoothedAggregation`
  ports amgcl's `plain_aggregates` (`_aggregates`: strong when
  `a_ij^2 > eps^2 a_ii a_jj`, eps halved per level, one serial greedy
  pass in row order, rows without strong neighbors left out, id `-1`)
  and its smoothed prolongation (`_smoothed_prolongation`: weak entries
  lumped into the diagonal, `-omega * (1 / dia)` as amgcl rounds it,
  rows sorted), equal to amgcl bit for bit (`tests/data/amgcl/`, C++
  driver there; keep it so). `omega = 4/3 / lambda_max(D^-1 A)` from
  `eigenvalue_range` (Gershgorin if CG breaks down); `R = P^T`,
  `A_c = R @ (A @ P)`; coarsest at `coarse_size` by `SparseCholesky`,
  regularized `SparseLDLT` if singular. V-cycle from zero: Chebyshev on
  `[0.1, 1.1] lambda` (Saad 12.1) or forward/backward Gauss-Seidel.
  Residuals by loops (`_residual`), not `RealArray`'s `-`.

### Parallel loops

`parallel_ranges(n, serial_below, body)` (`molina/parallel.mojo`) runs
`body(chunk, lo, hi)` on one chunk per logical core (per physical core
with `memory_bound=True`, which the matvec passes: a second hardware
thread speeds up arithmetic, not memory streams), or serially when
`n < serial_below`. The loops run on molina's own pool: POSIX threads
started on first use and kept, which poll for 100 µs after a loop and
then sleep on a futex; chunks are claimed by compare-and-swap on one
ticket (generation and next chunk), and the runtime's global teardown
after `main` joins the workers (`mojo run` frees their code next). It
uses two private runtime APIs, `std.ffi._get_global` (with a destroy
function) and `abi("C")` thread entry points, so check it on each new
nightly. A call costs 1–3 µs on polling workers and about 8 µs on
sleeping ones (MAX's `parallelize`, used before, 12–130 µs). A loop
started inside a body, or from a second thread while one runs, runs
serially. The thresholds (`PARALLEL_NNZ`, 1M stored entries, for the
matvec; 5e6 multiply-adds for the supernodal factor) date from MAX's cost
and are set where the serial loop takes a few hundred µs.
`MOLINA_NUM_THREADS=k` caps the chunks at k (1 runs everything serially,
for single-threaded benchmarks). The workers keep the default
floating-point environment, so subnormals are not flushed.

- Bodies must not raise: record failures per chunk and raise afterwards.
- Keep a hot serial loop out of a function that also holds the parallel
  branch (`CSRMatrix._matvec_parallel` is separate): in the same function
  the serial loop ran about 15% slower.
- Loops that sum into shared entries stay bit for bit serial by
  ownership: each chunk owns a block of outputs and adds only to those,
  in the serial order.
- Allocate a body's scratch with `chunk_scratch(n, fill)`, never a bare
  `List(length=...)`: two chunks' small lists can share a cache line, and
  writing them in an inner loop (false sharing) made a loop slower on
  two threads than on one.
- Large outputs the parallel loops write in full are created with
  `List(unsafe_uninit_length=n)`: a `fill` is a serial memset that also
  takes every page fault on one thread.

## Coding standards

- Use American spelling everywhere (code, comments, doc strings, docs,
  commit messages): "-ize", not "-ise" (`normalize`, `initialize`).
- Return a small struct with named fields (`EighResult`, `SolveResult`,
  `Inertia`, ...) rather than a tuple, and an enum-like struct with
  `comptime` members (`StopReason`, `FactorLayout`, `DirectMethod`)
  rather than a Bool or String flag.
- Public symbols get Google-style doc strings (`Args:`, `Parameters:`,
  `Returns:`, `Raises:`; see `docs/writing-docs.md`); `pixi run
  doc-check` must report none missing.
- Solvers are implemented from the published mathematics or ported from
  permissively licensed code (faer, amgcl: notices in `NOTICE`), never
  from GPL-licensed code; don't name GPL projects as sources.

## Testing

- Each test file ends with
  `TestSuite.discover_tests[__functions_in_module()]().run()`, so any
  `def test_*() raises` in the file is picked up automatically.
  TestSuite reports times in milliseconds.
- `tests/data/amd/` holds faer's AMD permutations and `tests/data/amgcl/`
  amgcl's aggregates and prolongations, with the drivers that produced
  them; `amd_ordering` and `SmoothedAggregation` must equal them exactly.
- The tests' matrices come from MojoDEC: `tests/data/matrices/*.csr`
  (Whitney-form matrices on Kuhn grids, written by MojoDEC's
  `tools/export_molina_fixtures.mojo`; format and names in the README
  there), read by `matrix(name)` in `tests/fixtures.mojo`, which test
  files import as `from fixtures import matrix`. Tests run from the
  repository root, where the data paths resolve. To add one, add its
  spec to the exporter and regenerate; don't hand-edit the files. The
  amd and amgcl data were computed from these exact matrices.
- `fixtures.uniform(rows, cols, seed)` gives reproducible random arrays
  (no NuMojo here).
