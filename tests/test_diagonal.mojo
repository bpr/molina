from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
)

from std.math import sqrt

from molina.diagonal import DiagonalOperator
from molina.sparse import CSRMatrix
from molina.utils import real_vector_from_list


def test_products() raises:
    var D = DiagonalOperator([1.0, 2.0, 3.0])
    var A = CSRMatrix.from_coo(
        3, 3, [0, 1, 2, 2], [1, 2, 0, 2], [1.0, 1.0, 1.0, 1.0]
    )
    # D @ A scales rows, A @ D scales columns
    var DA = D @ A
    var AD = A @ D
    var Dc = D.to_csr()
    var DA_ref = Dc @ A
    var AD_ref = A @ Dc
    for i in range(3):
        for j in range(3):
            assert_equal(DA[i, j], DA_ref[i, j])
            assert_equal(AD[i, j], AD_ref[i, j])
    assert_equal(DA.nnz(), A.nnz())
    # D @ D stays diagonal
    var DD = D @ D
    assert_equal(DD[2], 9.0)
    assert_equal(DD[0, 1], 0.0)
    # D @ x is elementwise
    var y = D @ real_vector_from_list([1.0, 1.0, 2.0])
    assert_equal(y.unsafe_get(2), 6.0)


def test_inverse_inner_product_norm() raises:
    var D = DiagonalOperator([4.0, 0.5])
    var Di = D.inverse()
    assert_equal(Di[0], 0.25)
    assert_equal((D @ Di)[1], 1.0)
    var x = real_vector_from_list([1.0, 2.0])
    var z = real_vector_from_list([3.0, -1.0])
    assert_equal(D.inner_product(x, z), 4.0 * 3.0 + 0.5 * -2.0)
    assert_equal(D.norm(x), sqrt(4.0 + 0.5 * 4.0))
    with assert_raises():
        _ = DiagonalOperator([1.0, 0.0]).inverse()


def test_arithmetic_and_errors() raises:
    var D = DiagonalOperator([1.0, -2.0])
    var E = 2.0 * D + D * 3.0 - (-D) / 2.0
    assert_equal(E[1], -2.0 * 5.5)
    assert_equal(len(DiagonalOperator.identity(4)), 4)
    var dense = D.todense()
    assert_equal(dense.unsafe_get(3), -2.0)
    with assert_raises():
        _ = D @ DiagonalOperator([1.0])
    with assert_raises():
        _ = D @ CSRMatrix.identity(3)
    with assert_raises():
        _ = D[0, 2]
    with assert_raises(contains="index 2 out of range"):
        _ = D[2]
    with assert_raises(contains="index -1 out of range"):
        D[-1] = 1.0


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
