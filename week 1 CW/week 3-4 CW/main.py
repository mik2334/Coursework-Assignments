import matrix_functions as mf
import time
import matplotlib
import matplotlib.pyplot as plt

N = 8
# set matrix size N as 2^3

A = mf.random_matrix(N, N)
B = mf.random_matrix(N, N)
# create 2 new random matrices 

C = mf.recursive_mat_mult(A, B)
# perform recursive matrix multiplication 
D = mf.strassen_mat_mult(A, B)
# perform strassen matrix multiplication 
E = mf.matrix_multiplication(A, B)
# perform naive matrix multiplication

mf.print_matrix(C)
mf.print_matrix(D)
mf.print_matrix(E)
# print all resultant matrices


### TASK 3
#################################################################################################################################################

methods = {"Naive": mf.matrix_multiplication, "Recursive": mf.recursive_mat_mult, "Strassen": mf.strassen_mat_mult}
# create dictionary to match names with each function
results = {name: [] for name in methods}
# create dictionary for results arrays
Ns = [2**i for i in range (1, 9)]
# create range for matrix size iteration
repetitions = 1

for N in Ns:
# create a while loop to test runtimes for different matrix sizes
    matrixA = mf.random_matrix(N, N)
    matrixB = mf.random_matrix(N, N)
    # create 2 random matrices of different set dimensions

    for name, func in methods.items():
    # for each function in the method dictionary
        start = time.time()
        # start the timer
        for _ in range(repetitions):
        # repeat however many times you choose
            func(matrixA, matrixB)
            # perform the function
        end = time.time()
        # end timer 
        avg_runtime = (end - start) / repetitions
        # calculate average runtime 
        results[name].append((N, avg_runtime))
        # append to each array in result dictionary 


plt.figure(dpi=100)

for name, data in results.items():
    # for each array in results dictionary
    Ns = [n for n, _ in data]
    times = [t for _, t in data]
    plt.plot(Ns, times, marker='o', label=name)
    # plot runtime against iteration size 

plt.xscale('log', base = 2)
plt.yscale('log')
# plot as a log-log graph, showing logarithmic scale 
plt.title("Runtime scaling against matrix multiplication iteration")
plt.xlabel("Matrix size, N (NxN)")
plt.ylabel("Average runtime, s")
plt.legend(title="Method")
plt.grid(True)

plt.show()

