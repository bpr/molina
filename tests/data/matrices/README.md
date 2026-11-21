# Matrices from MojoDEC

The tests' matrices from discrete exterior calculus: Whitney-form
matrices on Kuhn grids, built by MojoDEC and written by its
`tools/export_molina_fixtures.mojo`, which lists every file and what it
holds. `tests/fixtures.mojo`'s `matrix(name)` reads `<name>.csr` back
exactly as MojoDEC built it, explicit zeros included.

Names are `<kind>_<dims>`, for `kuhn_grid(dims)` (`hodge1_8x8` on
`kuhn_grid([8, 8])`):

- `d<k>`: the coboundary on k-simplices.
- `mass<k>`: the Whitney mass matrix `M_k`.
- `hodge<k>`: the Hodge stiffness plus mass, `M_k + d_k^T M_{k+1} d_k`
  (`M_n` alone at the top degree): symmetric positive definite.
- `p1`: the P1 Laplacian `d_0^T M_1 d_0` (Neumann, singular).
- `curlcurl`: `d_1^T M_2 d_1`.
- `<kind>_dirichlet`: rows and columns restricted to the interior
  simplices; `gradient_dirichlet` is `d_0` from the interior vertices to
  the interior edges.

## Format

Text, smaller than Matrix Market by 3.3 times, because a Kuhn grid's
matrices have only a few hundred distinct values:

```
# comment lines
nrows ncols nnz nvalues
value                      (nvalues lines: each distinct value once)
col id col id ...          (nrows lines: row i's entries in CSR order,
                            0-based column, index into the values)
```

Values are the shortest text that reads back to the same double
(integers without a decimal point); an empty line is an empty row.

## To regenerate

From `../MojoDEC`, after `pixi run numojo`:

```sh
pixi run mojo run -I . -I build tools/export_molina_fixtures.mojo \
    ../molina/tests/data/matrices
```

It reads each file back and checks every entry bit for bit. Regenerate
only if MojoDEC's Whitney matrices change on purpose: the amd and amgcl
reference data were computed from these exact matrices.
