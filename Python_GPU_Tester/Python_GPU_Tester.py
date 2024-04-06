import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
import time

# Rozmiar macierzy
matrix_x = 25000
matrix_y = 25000
matrix_size = (matrix_x, matrix_y)

# Generowanie losowych macierzy
print("Generowanie losowych macierzy...")
matrix_a = np.random.randn(*matrix_size).astype(np.float32)
matrix_b = np.random.randn(*matrix_size).astype(np.float32)

# Funkcja wykonywana na GPU (mnozenie macierzy)
matrix_mul = ElementwiseKernel(
    "float *a, float *b, float *c",
    "c[i] = a[i] * b[i]",
    "matrix_mul"
)

# Przygotowanie pamieci na GPU
print("Przygotowywanie pamieci na GPU...")
gpu_matrix_a = gpuarray.to_gpu(matrix_a.ravel())
gpu_matrix_b = gpuarray.to_gpu(matrix_b.ravel())
gpu_result_matrix = gpuarray.empty_like(gpu_matrix_a)

# Wykonanie mnozenia macierzy na GPU
start_time = time.time()
print("Czas rozpoczecia: {:.4f}".format(start_time))
for i in range(200):
    matrix_mul(gpu_matrix_a, gpu_matrix_b, gpu_result_matrix)
    gpu_result_matrix_host = gpu_result_matrix.get()
end_time = time.time()
print("Czas zakonczenia: {:.4f}".format(end_time))

# Obliczenie czasu wykonania
execution_time = end_time - start_time

print("Czas wykonania obliczen na GPU: {:.4f} sekund".format(execution_time))
time.sleep(10)