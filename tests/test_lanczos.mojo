from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from molina.diagonal import DiagonalOperator
from molina.lanczos import eigsh
from molina.linalg import eigh_generalized
from molina.real_array import RealArray
from molina.sparse import CSRMatrix
from molina.utils import dot, real_vector

from fixtures import matrix


def _worst(eta: RealArray) -> Float64:
    var w: Float64 = 0
    for i in range(eta.size):
        w = max(w, eta.unsafe_get(i))
    return w


def _dense_nearest(
    K: CSRMatrix, M: CSRMatrix, sigma: Float64, nev: Int
) raises -> List[Float64]:
    """The nev eigenvalues nearest sigma, ascending, by the dense solver."""
    var lam = eigh_generalized(K.todense(), M.todense()).values.copy()
    var taken = List[Bool](length=lam.size, fill=False)
    var out = List[Float64]()
    for _ in range(nev):
        var best = -1
        for i in range(lam.size):
            if not taken[i] and (
                best < 0
                or abs(lam.unsafe_get(i) - sigma)
                < abs(lam.unsafe_get(best) - sigma)
            ):
                best = i
        taken[best] = True
        out.append(lam.unsafe_get(best))
    sort(out)
    return out^


def test_smallest_eigenvalues_of_a_laplacian() raises:
    # the Neumann P1 Laplacian on a square: 0 is an eigenvalue, so the
    # shift sits just below it
    var K = matrix("p1_12x10")
    var M = matrix("mass0_12x10")
    var nev = 8
    var eig = eigsh(K, M, nev, sigma=-0.1)
    var want = _dense_nearest(K, M, -0.1, nev)
    assert_equal(eig.values.size, nev)
    for i in range(nev):
        assert_almost_equal(
            eig.values.unsafe_get(i),
            want[i],
            atol=1e-10 * (1 + want[i]),
            rtol=0,
        )
    assert_true(_worst(eig.residuals(K, M)) < 1e-13)
    # the eigenvectors are M-orthonormal
    for a in range(nev):
        var x = _column(eig.vectors, a, nev)
        for b in range(nev):
            var y = _column(eig.vectors, b, nev)
            assert_almost_equal(
                dot(x, M @ y), 1.0 if a == b else 0.0, atol=1e-10, rtol=0
            )


def _column(V: RealArray, j: Int, m: Int) raises -> RealArray:
    var n = V.size // m
    var x = real_vector(n)
    for i in range(n):
        x.unsafe_set(i, V.unsafe_get(i * m + j))
    return x^


def test_interior_shift_and_diagonal_mass() raises:
    # sigma inside the spectrum: K - sigma M is indefinite, and the nev
    # eigenvalues on both sides of it are found; then a Hodge star as the
    # mass, and the standard problem with the identity
    var K = matrix("p1_dirichlet_9x9")
    var M = matrix("mass0_dirichlet_9x9")
    for sigma in [150.0, 400.0]:
        var eig = eigsh(K, M, 5, sigma=sigma)
        var want = _dense_nearest(K, M, sigma, 5)
        for i in range(5):
            assert_almost_equal(
                eig.values.unsafe_get(i), want[i], atol=1e-9 * want[i], rtol=0
            )
        assert_true(_worst(eig.residuals(K, M)) < 1e-13)
    var I = CSRMatrix.identity(K.nrows)
    var std = eigsh(K, I, 4, sigma=0.0)
    var want = _dense_nearest(K, I, 0.0, 4)
    for i in range(4):
        assert_almost_equal(
            std.values.unsafe_get(i), want[i], atol=1e-10 * want[i], rtol=0
        )
    var diag = List[Float64]()
    for i in range(K.nrows):
        diag.append(M.todense().unsafe_get(i * K.nrows + i) * 2)
    var D = DiagonalOperator(diag^)
    var lumped = eigsh(K, D, 3, sigma=0.0)
    var wantd = _dense_nearest(K, D.to_csr(), 0.0, 3)
    for i in range(3):
        assert_almost_equal(
            lumped.values.unsafe_get(i), wantd[i], atol=1e-10 * wantd[i], rtol=0
        )
    assert_true(_worst(lumped.residuals(K, D)) < 1e-13)


def test_curl_curl_with_gradients_deflated() raises:
    # Maxwell's cavity on the unit square, tangential field zero on the
    # boundary: K = d1^T M2 d1 vanishes on the gradients d0 (one per
    # interior vertex). Deflating them, the first eigenvalues are the
    # nonzero ones, near pi^2 (1, 1, 2, 4, 4, ...)
    var K = matrix("curlcurl_dirichlet_10x10")
    var M = matrix("mass1_dirichlet_10x10")
    var G = matrix("gradient_dirichlet_10x10")
    var nev = 6
    # sigma = 0 is the kernel's eigenvalue, so K - 0 M is singular even
    # though the deflated iteration never meets the kernel: shift below it
    var eig = eigsh(K, M, nev, sigma=-1.0, deflate=G)
    var lam = eigh_generalized(K.todense(), M.todense()).values.copy()
    var nonzero = List[Float64]()
    for i in range(lam.size):
        if lam.unsafe_get(i) > 1e-8 and len(nonzero) < nev:
            nonzero.append(lam.unsafe_get(i))
    for i in range(nev):
        assert_almost_equal(
            eig.values.unsafe_get(i),
            nonzero[i],
            atol=1e-10 * nonzero[i],
            rtol=0,
        )
    assert_true(_worst(eig.residuals(K, M)) < 1e-13)
    # the modes are M-orthogonal to the gradients
    for a in range(nev):
        var x = _column(eig.vectors, a, nev)
        var g = G.transpose() @ (M @ x)
        for i in range(g.size):
            assert_almost_equal(g.unsafe_get(i), 0.0, atol=1e-10, rtol=0)


def test_small_and_exhausted_spaces() raises:
    # nev = n: the whole space; and a deflation leaving fewer than nev
    var diag: List[Float64] = [3.0, 1.0, 4.0, 1.5, 9.0]
    var K = DiagonalOperator(diag.copy()).to_csr()
    var I = CSRMatrix.identity(5)
    var all = eigsh(K, I, 5, sigma=0.0)
    var want: List[Float64] = [1.0, 1.5, 3.0, 4.0, 9.0]
    for i in range(5):
        assert_almost_equal(
            all.values.unsafe_get(i), want[i], atol=1e-13, rtol=0
        )
    # deflate e_0, e_1, e_2: only 2 pairs remain
    var r: List[Int] = [0, 1, 2]
    var c: List[Int] = [0, 1, 2]
    var v: List[Float64] = [1.0, 1.0, 1.0]
    var G = CSRMatrix.from_coo(5, 3, r, c, v)
    var two = eigsh(K, I, 2, sigma=0.0, deflate=G, ncv=4)
    assert_almost_equal(two.values.unsafe_get(0), 1.5, atol=1e-13, rtol=0)
    assert_almost_equal(two.values.unsafe_get(1), 9.0, atol=1e-13, rtol=0)
    with assert_raises(contains="only 2 eigenpairs exist"):
        _ = eigsh(K, I, 3, sigma=0.0, deflate=G, ncv=4)


def test_errors() raises:
    var diag: List[Float64] = [1.0, 2.0, 3.0]
    var K = DiagonalOperator(diag^).to_csr()
    var I = CSRMatrix.identity(3)
    with assert_raises(contains="does not factor at sigma = 2.0"):
        _ = eigsh(K, I, 1, sigma=2.0)
    with assert_raises(contains="nev = 4 for n = 3"):
        _ = eigsh(K, I, 4, sigma=0.5)
    with assert_raises(contains="ncv = 2 must be in"):
        _ = eigsh(K, I, 2, sigma=0.5, ncv=2)
    with assert_raises(contains="same square size"):
        _ = eigsh(K, CSRMatrix.identity(2), 1, sigma=0.5)
    # a rank-deficient G
    var r: List[Int] = [0, 1]
    var c: List[Int] = [0, 1]
    var v: List[Float64] = [1.0, 1.0]
    var G = CSRMatrix.from_coo(3, 3, r, c, v)
    with assert_raises(contains="full column rank"):
        _ = eigsh(K, I, 1, sigma=0.5, deflate=G)
    var minus = CSRMatrix.identity(3) * -1.0
    with assert_raises(contains="M is not positive definite"):
        _ = eigsh(K, minus, 1, sigma=0.5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
