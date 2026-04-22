NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 -lcusparse -lcublas

all: fp64_gmres_test.exe

fp64_gmres_test.exe: main.cu fp64_gmres.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f fp64_gmres_test.exe