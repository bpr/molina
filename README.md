# molina

Linear algebra in pure Mojo:

- `RealArray` vectors and dense matrices, and `CSRMatrix`.
- The `LinearOperator` and `Preconditioner` traits.
- Dense pivoted LU, Cholesky and Householder/QL `eigh`.
- AMD ordering, and simplicial and supernodal sparse LDLᵀ and Cholesky,
  the supernodal factor parallel.
- Krylov solvers: `cg`, `minres` and `lsqr`.
- Shift-invert Lanczos (`eigsh`).
- Smoothed-aggregation algebraic multigrid.

Its parallel loops run on its own POSIX-threads pool. molina imports
nothing but Mojo's standard library: no MAX, no NuMojo. It runs on Linux
x86-64.

It was split from [MojoDEC](https://github.com/bpr/MojoDEC), which builds
on it, as do mogeometry and minigrad. The sparse direct solvers port
[faer](https://codeberg.org/sarah-quinones/faer)'s and the multigrid
ports [amgcl](https://github.com/ddemidov/amgcl)'s aggregation; `NOTICE`
has their licenses.

```mojo
from molina import CSRMatrix, cg, jacobi, real_vector_from_list

def main() raises:
    var A = CSRMatrix.from_coo(
        2, 2, [0, 0, 1, 1], [0, 1, 0, 1], [4.0, 1.0, 1.0, 3.0]
    )
    var b = real_vector_from_list([1.0, 2.0])
    var x = cg(A, b, M=jacobi(A), rtol=1e-12).take_x()
```

## Using it from a pixi project

molina is a pixi-build package, which pixi compiles from source in your
project's environment. You need pixi 0.84 or newer (`pixi self-update`).
Add to your `pixi.toml`:

```toml
[workspace]
preview = ["pixi-build"]

[dependencies]
molina = { git = "https://github.com/bpr/molina", tag = "v0.1.0" }

# Compile molina with your project's Mojo: a .mojoc loads only in the
# compiler that built it, and pixi would otherwise use the newest one.
[workspace.build-variants]
mojo-compiler = ["1.3.0.dev2026110305"]   # the mojo version you use
```

Then `from molina import ...` works with no `-I`. Use `rev = "<commit>"`
or `branch = "main"` in place of `tag` to follow unreleased changes.

To work on molina and a project that uses it at once, point the
dependency at your checkout, `molina = { path = "../molina" }`, and
don't commit that.

## Developing

```sh
pixi run test        # tests/test_*.mojo
pixi run package     # build/molina.mojoc
pixi run doc-check   # every public symbol documented (docs/writing-docs.md)
pixi run docs        # API reference in docs/api/
pixi run format
```

`MOLINA_NUM_THREADS=k` caps the parallel loops at k threads, and
`MOLINA_NUM_THREADS=1` runs them serially.

## License

MIT (see `LICENSE`), with the faer, SuiteSparse AMD and amgcl notices in
`NOTICE`.
