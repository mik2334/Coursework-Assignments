import numpy as np
from numpy import random
import matplotlib
import matplotlib.pyplot as plt
import time

N = 1
M = 100
P = 100

results = []

def random_matrix(rows, columns):
# initialises a function to generate random numbers to fill the matrix
    if not(isinstance(rows, int) and isinstance(columns, int)):
        raise ValueError("Values must be of the integer type")
    # if the inputted rows and columns of not integers, output an error
    if rows <= 0 or columns <= 0:
        raise ValueError("Rows and columns must be a positive integer")
    # if the inputted rows and columns are less than 0, output an error
    if rows > 1000 or columns > 1000:
        raise ValueError("Matrix too large")
    # if the inputted rows and columns are too big, output an error 
    matrix = np.random.randint(100, size=(rows, columns))
    # uses numpy random number function to generate a matrix of inputted number of rows and columns
    # populates it with integers from 0 -> 999
    return(matrix)
    # prints resultant matrix when called

def matrix_multiplication(matrix1, matrix2):
    columnA = len(matrix1[0])
    # create a variable for columns in first input matrix
    rowB = len(matrix2)
    # create a variable for rows in second input matrix
    if columnA != rowB:
        raise ValueError("Incompatible dimensions for multiplication: ", columnA, " !=", rowB)
    # if the columns in the first matrix dont match the rows in the second matrix, output an error
    resultantMatrix = [[0 for x in range(len(matrix2[0]))] for y in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
              resultantMatrix[i][j] += int(matrix1[i][k]*matrix2[k][j])
    return(resultantMatrix)
# made a function to multiply two random matrices together so i can reuse the function 

while N < 150:
# create a while loop to test runtimes for different matrix sizes
    matrixA = random_matrix(M, N)
    matrixB = random_matrix(N, N)
    matrixC = random_matrix(N, P)
    # create 3 random matrices of different set dimensions
    # M and P are set numbers so we can test how N affects the runtime, as N affects the complexity the most as it is a polynomial

    start = time.time()
    # start the timer

    matrixX = matrix_multiplication(matrixB, matrixC)
    matrixD = matrix_multiplication(matrixX, matrixA)
    # create the intermediary and final matrix

    end = time.time()
    #end the timer
    runtime = end - start
    # calculate runtime

    results.append([N, runtime])
    # append iteration and runtime to the 2D array

    N += 1
    # iterate matrix size by 1x1

plt.figure(dpi=100)
x = [row[0] for row in results]
# for the 1st column, set that as the x axis of the plot
y = [row[1] for row in results]
# the second column is the y axis 

plt.plot(x,y)

plt.title("Runtime against matrix multiplication iteration")
plt.xlabel("N Iteration")
plt.ylabel("Runtime")
# plot the grapg of runtime per matrix iteration pl

plt.show()


    



