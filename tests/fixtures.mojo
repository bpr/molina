"""The tests' matrices: Whitney-form matrices on Kuhn grids, exported from
MojoDEC into `tests/data/matrices/` (see the README there)."""

from molina.real_array import RealArray
from molina.sparse import CSRMatrix
from molina.utils import real_array, real_vector


def matrix(name: String) raises -> CSRMatrix:
    """The matrix `tests/data/matrices/<name>.csr`, exactly as MojoDEC
    built it (explicit zeros included). Run the tests from the
    repository root."""
    var path = String("tests/data/matrices/", name, ".csr")
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var lines = text.split("\n")
    var k = 0
    while lines[k].startswith("#"):
        k += 1
    var head = lines[k].split()
    var nrows = atol(head[0])
    var ncols = atol(head[1])
    var nnz = atol(head[2])
    var nv = atol(head[3])
    var values = List[Float64](capacity=nv)
    for v in range(nv):
        values.append(atof(lines[k + 1 + v]))
    var indptr = List[Int](capacity=nrows + 1)
    var indices = List[Int](capacity=nnz)
    var data = List[Float64](capacity=nnz)
    indptr.append(0)
    var first = k + 1 + nv
    for i in range(nrows):
        var w = lines[first + i].split()
        for p in range(0, len(w), 2):
            indices.append(atol(w[p]))
            data.append(values[atol(w[p + 1])])
        indptr.append(len(indices))
    if len(indices) != nnz:
        raise Error(String(path, ": ", len(indices), " entries, not ", nnz))
    return CSRMatrix(nrows, ncols, indptr^, indices^, data^)


def uniform(rows: Int, cols: Int, seed: UInt64) raises -> RealArray:
    """A `rows x cols` array of uniform values in [0, 1) from a
    linear congruential generator; `cols = 0` gives a vector."""
    var a = real_vector(rows) if cols == 0 else real_array(rows, cols)
    var s = seed * 2862933555777941757 + 3037000493
    for i in range(a.size):
        s = s * 6364136223846793005 + 1442695040888963407
        a.unsafe_set(i, Float64(s >> 11) * (1.0 / 9007199254740992.0))
    return a^
