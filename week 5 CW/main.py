# import the doubleMatrix class in the same directory 
import doubleMatrix as dm
import time
import matplotlib.pyplot as plt

# create a function to test the block decomposition for different minimum sub-block sizes 
def runtime(matrix_size, n0_values, repeats):
    # generate two arrays using the doubleMatrix class 
    A = dm.doubleMatrix(matrix_size, matrix_size).random_matrix(0, 10)
    B = dm.doubleMatrix(matrix_size, matrix_size).random_matrix(0, 10)

    # create an array to store runtimes for each sub-block size 
    runtimes = []

    # for each sub-block size in the provided list
    for n0 in n0_values:
        times = []
        # for the number of input repeats 
        for _ in range(repeats):
            # time the runtime of the recursive matrix decomposition 
            start = time.time()
            A.matrix_recursive_mult(B, n0=n0)
            end = time.time()
            times.append(end - start)
        # append each runtime to the runtime list 
        runtimes.append(sum(times) / repeats)

    return runtimes


# create function to plot runtimes 
def plot_runtime(matrix_size, n0_values, repeats):
    runtimes = runtime(matrix_size, n0_values, repeats)
    
    plt.figure()
    plt.plot(n0_values, runtimes, marker='o')
    plt.xlabel("Minimum sub-block size, n0")
    plt.ylabel("Average runtime (seconds)")
    plt.title(f"Runtime vs sub-block size (n={matrix_size})")
    plt.grid(True)
    plt.show()


#set value of repeats
repeats = 3
# set minimum sub-block sizes
n0_values = [1, 2, 4, 8, 16]
# plot runtimes
plot_runtime(128, n0_values, repeats)
