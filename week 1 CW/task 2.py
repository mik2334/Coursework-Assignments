import numpy as np
from numpy import random
import matplotlib
import matplotlib.pyplot as plt
import time

### TASK 2
### part 1
############################################################################################################
N = int(input("Enter number of rows: "))
M = int(input("Enter number of columns: "))
# this asks the user for input values of rows and columns to create a matrix

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

print(random_matrix(N, M))
# prints a random matrix of size N x M 
############################################################################################################


### part 2
############################################################################################################
N = 5
# set N as an arbitrary number just to test input, can adjust to take user input if needed

matrixA = random_matrix(N, N)
print("Random matrix A is: \n", matrixA)
# create a random matrix A, using the above function, of size NxN and print it so i can see it properly 

matrixB = random_matrix(N, N)
print("Random matrix B is: \n", matrixB)
# create a random matrix B of size NxN and once again print visibly 

matrixC = [[0 for x in range(N)] for y in range(N)]
# create an matrix of the same size as the previous 2, all elements being 0

for i in range(len(matrixA)):
# for the number of rows in the first matrix 
    for j in range(len(matrixB[0])):
    # for the number of columns in second matrix 
        for k in range(len(matrixB)):
        # for the number of required multiplications (in this case N)
            matrixC[i][j] += int(matrixA[i][k]*matrixB[k][j])
            # perform multiplications of rows and columns in matrices A and B with the common N

print(matrixC)
# print the resultant matrix
############################################################################################################




### part 3
############################################################################################################
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

N = 1
# set the number of rows x columns of the square matrix as an initial value of 1
results = []
# create an empty array to hold calculated values for time for each iteration of matrix size NxN

while N < 99:
    matrixA = random_matrix(N, N)
    matrixB = random_matrix(N, N)
    # create two random  matrices of sizes NxN
    start = time.time()
    # start the timer 
    matrixC = matrix_multiplication(matrixA, matrixB)
    # perform matrix multiplication using premade function 
    end = time.time()
    # end timer
    runtime = end - start
    # total runtime for each operation 

    results.append([N, runtime])
    # append each runtime per each iteration to the 2D array
    N += 1
    # iterate the size of the matrix
############################################################################################################
    


### part 4
############################################################################################################
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
############################################################################################################

    
            
