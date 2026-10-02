""" MA3.py

Student: Moa Lundin
Mail: moa.lundin.2633@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from numba import njit
from statistics import mean 
from time import perf_counter as pc

# Exc1
def approximate_pi(n):
    # n is the number of points

    # Write your code here
    print(f'The number of points n: {n}')

    # Creates n points
    points = [(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(n)]

    # Check which points are in and which are outside the circle
    inside = []
    outside = []


    for x, y in points:
        if x**2 + y**2 <= 1:
            inside.append((x, y))
        else:
            outside.append((x, y))

    # n = n_c + n_s
    # n_c are number of points within the circle
    n_c = len(inside)

    # Plot the points and store as a png file
    plt.figure(figsize=(6, 6))
    if inside:
        x_in, y_in = zip(*inside)
        plt.scatter(x_in, y_in, color='red')

    if outside:
        x_out, y_out = zip(*outside)
        plt.scatter(x_out, y_out, color='blue')

    plt.title('Approximation of π using Monte Carlo')
    plt.text(0.0, -0.18, f'The approximation of π ≈ {4*n_c/n}',          
    transform=plt.gca().transAxes, fontsize=9,                                
    verticalalignment='top',
    horizontalalignment='left',               
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()

    #plt.savefig(f'Exc1_n={n}')
    plt.show()

    print(f'The approximation of π ≈ {4*n_c/n}\n')
    return 4*n_c/n


# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points

    # d is the number of dimensions of the sphere
    
    # Creates n points
    points = [[random.uniform(-1,1) for _ in range(d)] for _ in range(n)]
    
    # Check which points are in and which are outside the sphere
    distances = list(map(lambda p: sum(x**2 for x in p), points))
    inside = list(filter(lambda x: x <= 1, distances))
        
    # Approximate V_d(1)
    Vd = (len(inside)*2**d)/n
    print(f'The approximation of V_d(1) = {Vd}')

    return Vd

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points
    # d is the number of dimensions of the sphere 

    Vd = lambda d : (m.pi**(d/2))/(m.gamma(d/2 + 1))

    return Vd(d)

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points

    # d is the number of dimensions of the sphere
    #np is the number of processes

    inside = 0
    # Creates n points
    for i in range(n):
        distance = 0
        for j in range(d):
            x = random.uniform(-1,1)
            distance += x**2
        if distance <= 1:
            inside += 1

    Vd = (inside*2**d)/n
    print('The approximation of V_d(1) = ', Vd)

    return Vd

#Exc4: parallel code - parallelize actual computations by splitting data
def worker(p, d):
        # Creates n points
        points = [[random.uniform(-1,1) for _ in range(d)] for _ in range(p)]
        
        # Check which points are in and which are outside the sphere
        distances = list(map(lambda k: sum(x**2 for x in k), points))
        inside = list(filter(lambda x: x <= 1, distances))

        return len(inside)

def sphere_volume_parallel(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes

    p_in = 0
    with future.ProcessPoolExecutor(np) as ex:
        p = [int(n/np) for i in range(np)]
        d_lst = [d for i in range(np)]
        results = ex.map(worker, p, d_lst)

        for r in results:
            p_in += r

    Vd = (p_in*2**d)/n

    return Vd
    
def main():
    # Exc1
    print('Exc1:\n')
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    print('\nExc2:\n')
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}\n")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}\n")

    # Exc3
    print("Exc3: ")
    n = 1000000
    d = 11
    for i in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        print(f"Sequential time of {d} and {n}: {stop-start}, run {i+1}\n")
    print("What is numba time?")
    for i in range(3):
            start = pc()
            sphere_volume_numba(n, d)
            stop = pc()
            print(f"Sequential time of {d} and {n}: {stop-start}, run {i+1}\n")

    # Exc4
    print("\nExc4: \n")
    n = 1000000
    d = 11

    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Sequential time of {d} and {n}: {stop-start}\n")

    print("What is parallel time?")
    start = pc()
    sphere_volume_parallel(n, d)
    stop = pc()
    print(f"Parallel time of {d} and {n}: {stop-start}\n")

    
    

if __name__ == '__main__':
	main()
