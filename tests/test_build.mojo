"""Smoke tests: the 1D Laplacian solved by CG and by sparse Cholesky, and
a parallel loop covering its range."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from molina import (
    CSRMatrix,
    SparseCholesky,
    StopReason,
    cg,
    jacobi,
    real_vector,
)
from molina.parallel import parallel_ranges


def _laplacian_1d(n: Int) raises -> CSRMatrix:
    """tridiag(-1, 2, -1), `n x n`."""
    var rows = List[Int]()
    var cols = List[Int]()
    var vals = List[Float64]()
    for i in range(n):
        rows.append(i)
        cols.append(i)
        vals.append(2.0)
        if i > 0:
            rows.append(i)
            cols.append(i - 1)
            vals.append(-1.0)
            rows.append(i - 1)
            cols.append(i)
            vals.append(-1.0)
    return CSRMatrix.from_coo(n, n, rows, cols, vals)


def test_laplacian_solves() raises:
    # -u'' = 1 on (0, 1), u(0) = u(1) = 0: second differences are exact on
    # the quadratic u = x (1 - x) / 2, so the discrete solution is it.
    var n = 50
    var h = 1.0 / Float64(n + 1)
    var A = _laplacian_1d(n)
    var b = real_vector(n)
    for i in range(n):
        b.unsafe_set(i, h * h)
    var res = cg(A, b, M=jacobi(A), rtol=1e-14)
    assert_true(res.stop == StopReason.CONVERGED)
    var x_cg = res^.take_x()
    var x_ch = SparseCholesky(A).solve(b)
    for i in range(n):
        var t = Float64(i + 1) * h
        var u = t * (1.0 - t) / 2.0
        assert_almost_equal(x_cg.unsafe_get(i), u, atol=1e-12, rtol=0)
        assert_almost_equal(x_ch.unsafe_get(i), u, atol=1e-14, rtol=0)


def test_parallel_ranges_cover() raises:
    var n = 100_003
    var hits = List[Int](length=n, fill=0)
    var p = hits.unsafe_ptr()

    def body(c: Int, lo: Int, hi: Int) {p}:
        for i in range(lo, hi):
            p[unsafe_offset=i] += 1

    for _ in range(10):
        parallel_ranges(n, 0, body)
    var wrong = 0
    for i in range(n):
        if hits[i] != 10:
            wrong += 1
    assert_equal(wrong, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
