nvcc -c gpu_ray_surface_intersect.cu -o gpu_ray_surface_intersect.o
nvcc -c scm_cuda.cu -o scm_cuda.o
nvcc gpu_ray_surface_intersect.o scm_cuda.o -o scm
