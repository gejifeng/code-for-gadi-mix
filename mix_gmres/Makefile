NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 -lcusparse -lcublas

all: mix_gmres_test.exe

NVCC = /usr/local/cuda/bin/nvcc

mix_gmres_test.exe: main.cu mix_gmres.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f mix_gmres_test.exe
