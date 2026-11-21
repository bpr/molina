# Writing doc strings

The [API reference](api/index.md) is generated from the doc strings, so
they are the documentation. Every public module, struct, trait, field,
alias, and function needs one.

## Format

Google style, as Mojo's standard library uses. The first paragraph is the
summary; then any longer description, in Markdown; then the sections
that apply:

```mojo
def cholesky_nonzeros(A: CSRMatrix, perm: List[Int]) raises -> Int:
    """The number of nonzeros `SparseLDLT(A, perm=perm)` would store in
    `L` (its diagonal included), from the elimination tree alone.

    Args:
        A: An `n x n` matrix with a symmetric pattern.
        perm: The ordering: row `perm[k]` of `A` becomes row `k`.

    Returns:
        `nnz(L)`, as `factor_nonzeros()` would report it.

    Raises:
        If `A` is not square or `perm` is not a permutation of `0..n-1`.
    """
```

- **`Parameters:`** documents compile-time parameters, on the struct or
  function that declares them (`parallel_ranges[F]`).
- **`Args:`** lists every argument except `self`.
- **`Returns:`** is needed whenever the function returns a value.
- **`Raises:`** is needed whenever the function is `raises`.
- Fields and `comptime` aliases take a one-line doc string below them.
- A module's doc string starts with its title, underlined with `=`; the
  generator uses the title in the index.

Code in doc strings is not compiled or run (Mojo has no doctest runner),
so keep examples short and check them by hand; the tests and the examples
are the checked usage.

## Checking

```sh
pixi run doc-check                 # fail on any warning
pixi run doc-check --list sparse   # the warnings in matching files
```

`mojo doc --diagnose-missing-doc-strings` warns about missing doc strings
and missing sections; `doc-check` fails on any of its warnings, so
document new code as you write it.

## Generating

```sh
pixi run docs         # Markdown pages in docs/api/ (not committed)
```

`docs/gen_api.py` renders `mojo doc`'s JSON: a page per module, with
signatures, the sections above, and links between molina's types.
