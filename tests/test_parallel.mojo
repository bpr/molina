"""The parallel loops: the pool covers each range once and honors
MOLINA_NUM_THREADS, the matvec's parallel rows sum as the serial loop
does, and the supernodal factor and block solves give the serial bits at
every thread count."""

from std.os import setenv, unsetenv
from std.sys import num_logical_cores, num_physical_cores
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
)

from molina.ordering import amd_ordering
from molina.parallel import num_chunks, parallel_ranges
from molina.sparse import CSRMatrix, PARALLEL_NNZ
from molina.sparse_ldlt import FactorLayout, SparseLDLT
from molina.utils import fget, fset, real_array, real_vector

from fixtures import matrix


def test_parallel_ranges_covers_once() raises:
    var n = 100_003
    var hits = List[Int](length=n, fill=0)
    var hp = hits.unsafe_ptr()
    var chunks = num_chunks(n, 0)
    var seen = List[Int](length=chunks, fill=0)
    var sp = seen.unsafe_ptr()

    def body(chunk: Int, lo: Int, hi: Int) {hp, sp}:
        sp[unsafe_offset=chunk] += 1
        for i in range(lo, hi):
            hp[unsafe_offset=i] += 1

    parallel_ranges(n, 0, body)
    for h in hits:
        assert_equal(h, 1)
    for c in seen:
        assert_equal(c, 1)
    # below the threshold: one chunk on the calling thread
    assert_equal(num_chunks(10, 100), 1)
    assert_equal(num_chunks(0, 0), 0)


def test_num_threads_env_caps_chunks() raises:
    var cores = num_logical_cores()
    _ = setenv("MOLINA_NUM_THREADS", "1")
    assert_equal(num_chunks(100_000, 0), 1)
    _ = setenv("MOLINA_NUM_THREADS", "2")
    assert_equal(num_chunks(100_000, 0), min(2, cores))
    # not a positive integer: ignored
    _ = setenv("MOLINA_NUM_THREADS", "zero")
    assert_equal(num_chunks(100_000, 0), cores)
    _ = setenv("MOLINA_NUM_THREADS", "0")
    assert_equal(num_chunks(100_000, 0), cores)
    _ = unsetenv("MOLINA_NUM_THREADS")
    assert_equal(num_chunks(100_000, 0), cores)
    # memory-bound loops: one chunk per physical core, capped the same way
    var physical = num_physical_cores()
    assert_equal(num_chunks(100_000, 0, memory_bound=True), physical)
    _ = setenv("MOLINA_NUM_THREADS", "2")
    assert_equal(num_chunks(100_000, 0, memory_bound=True), min(2, physical))
    _ = unsetenv("MOLINA_NUM_THREADS")


def _grid_laplacian(n: Int) raises -> CSRMatrix:
    """The graph Laplacian of an n x n grid of vertices cut into triangles
    (edges right, up, and up-right): 7 entries per interior row."""
    var rows = List[Int]()
    var cols = List[Int]()
    var vals = List[Float64]()
    for i in range(n):
        for j in range(n):
            var v = i * n + j
            for step in [(0, 1), (1, 0), (1, 1)]:
                var a = i + step[0]
                var b = j + step[1]
                if a < n and b < n:
                    var w = a * n + b
                    rows.append(v)
                    cols.append(w)
                    vals.append(-1.0)
                    rows.append(w)
                    cols.append(v)
                    vals.append(-1.0)
                    rows.append(v)
                    cols.append(v)
                    vals.append(1.0)
                    rows.append(w)
                    cols.append(w)
                    vals.append(1.0)
    return CSRMatrix.from_coo(n * n, n * n, rows, cols, vals)


def test_large_matvec() raises:
    var A = _grid_laplacian(400)
    assert_true(A.nnz() >= PARALLEL_NNZ)
    var x = real_vector(A.ncols)
    for i in range(x.size):
        x.unsafe_set(i, Float64((i * 7) % 13) - 6.0)
    var y = A @ x
    # the same sums as a serial loop over the rows
    for i in range(A.nrows):
        var s: Float64 = 0
        for k in range(A.indptr[i], A.indptr[i + 1]):
            s += A.data[k] * x.unsafe_get(A.indices[k])
        assert_equal(y.unsafe_get(i), s)


# -- supernodal LDL^T: subtrees and pieces of large supernodes as tasks


# -- supernodal LDL^T: subtrees and pieces of large supernodes as tasks


def _laplacian_3d(N: Int, k: Int) raises -> CSRMatrix:
    """The Whitney Hodge stiffness plus mass on k-forms on an N^3 Kuhn
    grid."""
    return matrix(String("hodge", k, "_", N, "x", N, "x", N))


def _supernodal_values(
    A: CSRMatrix, perm: List[Int], signs: List[Int], threads: String
) raises -> List[Float64]:
    """The factor's blocks, then its number of regularized pivots."""
    if threads:
        _ = setenv("MOLINA_NUM_THREADS", threads)
    var s: Optional[List[Int]] = None
    if len(signs) > 0:
        s = signs.copy()
    var F = SparseLDLT(
        A,
        perm=perm.copy(),
        signs=s,
        regularization=1e-10 if len(signs) > 0 else 0.0,
        refinement_steps=0,
        layout=FactorLayout.SUPERNODAL,
    )
    _ = unsetenv("MOLINA_NUM_THREADS")
    var out = F._super.value().vals.copy()
    out.append(Float64(F.num_regularized))
    return out^


def _check_supernodal_matches_serial(
    A: CSRMatrix, signs: List[Int] = List[Int]()
) raises:
    var perm = amd_ordering(A)
    var want = _supernodal_values(A, perm, signs, "1")
    for threads in ["2", "3", "8", ""]:
        var got = _supernodal_values(A, perm, signs, threads)
        assert_equal(len(got), len(want))
        for i in range(len(want)):
            # bit for bit: each entry takes the same operations in order
            assert_equal(got[i], want[i])


def test_parallel_supernodal_matches_serial() raises:
    # past the work at which the factorization goes parallel; 12^3 and the
    # 1-forms have supernodes split into column windows and row tasks
    _check_supernodal_matches_serial(_laplacian_3d(12, 0))
    _check_supernodal_matches_serial(_laplacian_3d(7, 1))
    _check_supernodal_matches_serial(matrix("hodge0_90x90"))
    # a saddle point [[0, B], [B^T, -M]] with signs and regularization
    var M = matrix("mass1_6x6x6")
    var B = matrix("d1_6x6x6")
    var c = B.nrows
    var rows = List[Int]()
    var cols = List[Int]()
    var vals = List[Float64]()
    for i in range(M.nrows):
        for t in range(M.indptr[i], M.indptr[i + 1]):
            rows.append(c + i)
            cols.append(c + M.indices[t])
            vals.append(-M.data[t])
    for i in range(c):
        for t in range(B.indptr[i], B.indptr[i + 1]):
            rows.append(i)
            cols.append(c + B.indices[t])
            vals.append(B.data[t])
            rows.append(c + B.indices[t])
            cols.append(i)
            vals.append(B.data[t])
    var K = CSRMatrix.from_coo(c + M.nrows, c + M.nrows, rows, cols, vals)
    var signs = List[Int]()
    for _ in range(c):
        signs.append(1)
    for _ in range(M.nrows):
        signs.append(-1)
    _check_supernodal_matches_serial(K, signs)


def test_parallel_block_solves_match_serial() raises:
    # many right-hand sides: blocks of columns on separate threads, the
    # same bits as on one, and each column (refinement residuals included)
    # the single solve's
    var A = _laplacian_3d(12, 0)
    var n = A.nrows
    var m = 40
    var B = real_array(n, m)
    for c in range(m):
        for i in range(n):
            fset(B, i, c, Float64((i * 7 + c * 13) % 11) - 5.0)
    for layout in [FactorLayout.SIMPLICIAL, FactorLayout.SUPERNODAL]:
        var F = SparseLDLT(A, layout=layout, refinement_steps=2)
        _ = setenv("MOLINA_NUM_THREADS", "1")
        var want = F.solve(B)
        for c in range(0, m, 13):
            var b = real_vector(n)
            for i in range(n):
                b.unsafe_set(i, fget(B, i, c))
            var x = F.solve(b)
            for i in range(n):
                assert_equal(x.unsafe_get(i), fget(want, i, c))
        for threads in ["3", ""]:
            if threads:
                _ = setenv("MOLINA_NUM_THREADS", threads)
            else:
                _ = unsetenv("MOLINA_NUM_THREADS")
            var got = F.solve(B)
            for i in range(n * m):
                assert_equal(got.unsafe_get(i), want.unsafe_get(i))
        _ = unsetenv("MOLINA_NUM_THREADS")


def _supernodal_error(A: CSRMatrix, perm: List[Int], threads: String) -> String:
    if threads:
        _ = setenv("MOLINA_NUM_THREADS", threads)
    var message = String("")
    try:
        _ = SparseLDLT(A, perm=perm.copy(), layout=FactorLayout.SUPERNODAL)
    except e:
        message = String(e)
    _ = unsetenv("MOLINA_NUM_THREADS")
    return message^


def _check_supernodal_error(A: CSRMatrix, perm: List[Int]) raises:
    var want = _supernodal_error(A, perm, "1")
    assert_true("pivot" in want, want)
    for threads in ["2", "8", ""]:
        assert_equal(_supernodal_error(A, perm, threads), want)


def test_parallel_supernodal_failures_match_serial() raises:
    # a failed pivot stops every thread, and the error is the one the
    # serial order meets first
    var A = _laplacian_3d(12, 0)
    var n = A.nrows
    var v = n // 2
    # vertex v's row and column at 1e300, eliminated last: its pivot
    # overflows to -inf in the root supernode
    var rows = List[Int]()
    var cols = List[Int]()
    var vals = List[Float64]()
    for i in range(n):
        for t in range(A.indptr[i], A.indptr[i + 1]):
            var j = A.indices[t]
            rows.append(i)
            cols.append(j)
            vals.append(1e300 if i == v or j == v else A.data[t])
    var big = CSRMatrix.from_coo(n, n, rows, cols, vals)
    var perm = List[Int]()
    var amd = amd_ordering(A)
    for k in range(n):
        if amd[k] != v:
            perm.append(amd[k])
    perm.append(v)
    _check_supernodal_error(big, perm)
    # two isolated rows without a diagonal: zero pivots in subtrees of
    # their own
    for t in range(len(vals)):
        if rows[t] == v or cols[t] == v:
            vals[t] = A.data[t]  # from_coo kept the rows in order
    var Z = CSRMatrix.from_coo(n + 2, n + 2, rows, cols, vals)
    _check_supernodal_error(Z, amd_ordering(Z))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
