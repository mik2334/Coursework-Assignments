import numpy as np
from numpy import random

def random_matrix(rows, columns):
# reusing my function from the previous week coursework to generate a random matrix based on input sizes
    if not(isinstance(rows, int) and isinstance(columns, int)):
        raise ValueError("Values must be of the integer type")
    if rows <= 0 or columns <= 0:
        raise ValueError("Rows and columns must be a positive integer")
    if rows > 1000 or columns > 1000:
        raise ValueError("Matrix too large")
    matrix = np.random.randint(100, size=(rows, columns))
    return(matrix)
    
def matrix_multiplication(matrix1, matrix2):
# reusing my matrix multiplication function for last weeks coursework 
    columnA = len(matrix1[0])
    rowB = len(matrix2)
    if columnA != rowB:
        raise ValueError("Incompatible dimensions for multiplication: ", columnA, " !=", rowB)
    resultantMatrix = [[0 for x in range(len(matrix2[0]))] for y in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
              resultantMatrix[i][j] += int(matrix1[i][k]*matrix2[k][j])
    return(resultantMatrix)

### TASK 1
#################################################################################################################################################

def recursive_mat_mult(matrix1, matrix2):
    columnA = len(matrix1[0])
    rowB = len(matrix2)
    if columnA != rowB:
        raise ValueError("Incompatible dimensions for multiplication: ", columnA, " !=", rowB)
# create function to perform recursive matrix multiplication using divide and conquer 
    n = len(matrix1)
    # set length of matrix
    if(n==1):
    # if the size is 1x1
        return np.array([matrix1[0, 0] * matrix2[0,0]])
    # return the scalar multiplication 
    else:
        
        mid = n//2
        # set a midpoint
        
        A11 = matrix1[:mid, :mid]
        A12 = matrix1[:mid, mid:]
        A21 = matrix1[mid:, :mid]
        A22 = matrix1[mid:, mid:]
        
        B11 = matrix2[:mid, :mid]
        B12 = matrix2[:mid, mid:]
        B21 = matrix2[mid:, :mid]
        B22 = matrix2[mid:, mid:]
        # create all submatrices for matrix A and B using quadrants

        
        M1 = recursive_mat_mult(A11, B11)
        M2 = recursive_mat_mult(A11, B12)
        M3 = recursive_mat_mult(A21, B11)
        M4 = recursive_mat_mult(A21, B12) 
        M5 = recursive_mat_mult(A12, B21)
        M6 = recursive_mat_mult(A12, B22)
        M7 = recursive_mat_mult(A22, B21)
        M8 = recursive_mat_mult(A22, B22)
        # create the M blocks by recursively calling the function to create the block matrix multiplication

        C11 = M1 + M5
        C12 = M2 + M6 
        C21 = M3 + M7
        C22 = M4 + M8
        # create the C blocks by adding each M block to its corresponding block

        C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
        # add each submatrix of C into the resultant matrix C using numpy
        return C

### TASK 2
#################################################################################################################################################

def strassen_mat_mult(matrix1, matrix2):
# create strassen matrix multiplication funciton, follows similar logic to previous function 
    n = len(matrix1)
    # set length of matrix
    if(n==1):
    # if the size is 1x1
        return np.array([[matrix1[0, 0] * matrix2[0,0]]])
    
    mid = n//2
    # set a midpoint
    
    A11 = matrix1[:mid, :mid]
    A12 = matrix1[:mid, mid:]
    A21 = matrix1[mid:, :mid]
    A22 = matrix1[mid:, mid:]
    
    B11 = matrix2[:mid, :mid]
    B12 = matrix2[:mid, mid:]
    B21 = matrix2[mid:, :mid]
    B22 = matrix2[mid:, mid:]
    # create all submatrices for matrix A and B using quadrants

    P1 = strassen_mat_mult(A11, B12 - B22)
    P2 = strassen_mat_mult(A11 + A12, B22)
    P3 = strassen_mat_mult(A21 + A22, B11)
    P4 = strassen_mat_mult(A22, B21 - B11)
    P5 = strassen_mat_mult(A11 + A22, B11 + B22)
    P6 = strassen_mat_mult(A12 - A22, B21 + B22)
    P7 = strassen_mat_mult(A11 - A21, B11 + B12)
    # recursively call function using sum/difference of products of input matrices

    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7
    # construct resultant matrix C using products

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    # add each submatrix of C into the resultant matrix C using numpy
    return C

def print_matrix(matrix):
# create a new function to print the matrix in a readable manner 
    for i in range(len(matrix)):
    # for the number of rows in input matrix
        for j in range(len(matrix[i])):
        # for the number of columns in each row  of input matrix 
            print(matrix[i][j], end=' ')
            # print each column per row  
        print()
    print("\n")
    # print a new line after each matrix is printed to separate 
  
