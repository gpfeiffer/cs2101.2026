"""vectors and matrices, as in NB4, with the matrix product added

Run the examples below as tests with  python3 -m doctest tensor.py

>>> v = Vector([1, 2, 3])
>>> ma = Matrix([[1, 0, 1], [2, 1, 1], [0, 1, 1]])
>>> v + v, 2 * v, v - v
(Vector([2, 4, 6]), Vector([2, 4, 6]), Vector([0, 0, 0]))
>>> v.dot(v)
14
>>> ma.T
Matrix([Vector([1, 2, 0]), Vector([0, 1, 1]), Vector([1, 1, 1])])
>>> ma.T.T == ma
True
>>> v == [1, 2, 3], Vector([]) == Matrix([])
(False, False)
>>> v @ ma
Vector([5, 5, 6])
>>> ma @ ma
Matrix([Vector([1, 1, 2]), Vector([4, 2, 4]), Vector([2, 2, 2])])
>>> (ma @ ma).T == ma.T @ ma.T
True
>>> Matrix([[1, 2], [3]]).T
Traceback (most recent call last):
...
ValueError: zip() argument 2 is shorter than argument 1
"""


class Tensor:
    """a common parent class for vectors and matrices"""

    # constructor
    def __init__(self, data):
        self.data = list(data)

    # string representation
    def __repr__(self):
        return f"{type(self).__name__}({self.data})"

    # list like behaviour
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __eq__(self, other):
        return type(self) is type(other) and self.data == other.data

    # number like behaviour
    def __add__(self, other):
        assert len(self) == len(other), "length mismatch"
        return type(self)([x + y for x, y in zip(self, other)])

    def __rmul__(self, other):
        return type(self)([other * x for x in self])

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + -other


class Vector(Tensor):
    """a class for vectors, with dot product and vector-matrix product"""

    def dot(self, other):  # v . w
        assert len(self) == len(other), "length mismatch"
        return sum(x * y for x, y in zip(self, other))

    def __matmul__(self, other):  # v @ m
        return Vector([self.dot(x) for x in other.T])


class Matrix(Tensor):
    """a class for matrices, with transpose and matrix product"""

    def __init__(self, data):
        super().__init__(Vector(row) for row in data)

    @property
    def T(self):
        return Matrix(zip(*self, strict=True))

    def __matmul__(self, other):  # m @ m
        return Matrix([x @ other for x in self])
