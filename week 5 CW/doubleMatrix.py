import numpy as np

class doubleMatrix:

    # initialise MxN matrix stored as 1D numpy array
    def __init__(self, M, N, data=None):
        self.rows = M
        self.columns = N

        # allocate storage
        if data is None:
            # if no data is inputted when matrix initialised, set all elements to 0
            self.data = np.zeros(M*N, dtype=np.float64)
        else:
            # otherwise set provided data as a double array
            self.data = np.array(data, dtype=np.float64)
            assert self.data.size == M * N

    def idx(self, i, j):
        # convert 2D indices, i and j, into row-major indexing:
        # index = i * N + j
        return i * self.columns + j

    def get(self, i, j):
        # retrieve value from input index using idx method
        return self.data[self.idx(i, j)]

    def set(self, i, j, value):
        # set value in array using idx method
        self.data[self.idx(i, j)] = value

    # method to initialise 1D matrix array from input list of elements
    def init_from_list(self, elements):
        # ensure the number of elements in the list fits into the 1D array
        assert len(elements) == self.rows * self.columns
        # set input elements as elements of 1D array matrix, set data type to double
        self.data = np.asarray(elements, dtype=np.float64).copy()
        return self

    # method to generate zero matrix stored as 1D array
    def zero_matrix(self):
        self.data.fill(0.0)
        return self

    # method to generate uniform matrix of random arrays, with element values between a < x < b
    def random_matrix(self, a, b):
        self.data = np.random.uniform(a, b, self.rows * self.columns)
        return self

    # method to reshape matrix if dimensions allow
    def reshape(self, M, N):
        # if the total number of elements match  
        assert M * N == self.rows * self.columns, "shape must preserve total size"
        self.rows, self.columns = M, N
        # change the current dimensions to input dimensions 
        return self

    # method to transpose matrix
    def transpose(self):
        # create a temporary 2D array 
        A = self.data.reshape(self.rows, self.columns).T
        return doubleMatrix(A.shape[0], A.shape[1], A.flatten())
        
    # method to sum two input matrices 
    def matrix_sum(self, other):
        assert self.rows == other.rows and self.columns == other.columns
        return doubleMatrix(self.rows, self.columns, self.data + other.data)

    # method to subtract two input matrices 
    def matrix_subtract(self, other):
        assert self.rows == other.rows and self.columns == other.columns
        return doubleMatrix(self.rows, self.columns, self.data - other.data)

    # simple naive multiplication using previous assignments function
    # as well as object get and set methods
    def naive_mult(self, other):
        assert self.columns == other.rows
        A = self.data.reshape(self.rows, self.columns)
        B = other.data.reshape(other.rows, other.columns)
        
        C = np.zeros((self.rows, other.columns), dtype=np.float64)
        for i in range(self.rows):
            for j in range(other.columns):
                C[i,j] = np.dot(A[i, :], B[:, j])
        return doubleMatrix(self.rows, other.columns, C.flatten())

    # method to perform recursive matrix decomposition, based on the previous week assignment 
    def matrix_recursive_mult(self, other, n0):
        assert self.rows == self.columns == other.rows == other.columns

        A = self.data.reshape(self.rows, self.columns)
        B = other.data.reshape(other.rows, other.columns)

        def rec_mult(A, B):
            n = A.shape[0]

            if(n<=n0):
                return A @ B
            else:
                mid = n//2
                # set a midpoint
                
                A11 = A[:mid, :mid]
                A12 = A[:mid, mid:]
                A21 = A[mid:, :mid]
                A22 = A[mid:, mid:]
                
                B11 = B[:mid, :mid]
                B12 = B[:mid, mid:]
                B21 = B[mid:, :mid]
                B22 = B[mid:, mid:]
                # create all submatrices for matrix A and B using quadrants

                C11 = rec_mult(A11, B11) + rec_mult(A12, B21)
                C12 = rec_mult(A11, B12) + rec_mult(A12, B22) 
                C21 = rec_mult(A21, B11) + rec_mult(A22, B21)
                C22 = rec_mult(A21, B12) + rec_mult(A22, B22)
                # create the C blocks by adding each M block to its corresponding block
                # removed the storing of M blocks to improve memory efficiency 

                return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

        C = rec_mult(A, B)

        return doubleMatrix(self.rows, other.columns, C.flatten())
        
    # formatted print output
    def __str__(self):
        A = self.data.reshape(self.rows, self.columns)
        return "\n".join(" ".join(f"{A[i,j]:.3f}" for j in range(self.columns))
                         for i in range(self.rows)) + "\n"
    

"""
matrix1 = doubleMatrix(4, 4).random_matrix(0, 10)
print(matrix1)

elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
matrix2 = doubleMatrix(4, 4).init_from_list(elements)
print(matrix2)

matrix3 = matrix1.matrix_recursive_mult(matrix2)
print(matrix3)

matrix4 = doubleMatrix(4, 4)
print(matrix4)
"""
