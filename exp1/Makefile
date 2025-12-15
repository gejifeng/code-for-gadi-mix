# Makefile for CUDA Sylvester Solver

# Compiler and flags
NVCC = /usr/local/cuda/bin/nvcc
CXXFLAGS = -std=c++17 -O3 -I/usr/local/cuda/include
CUDAFLAGS = -arch=sm_80 -lcublas -lcusparse -L/usr/local/cuda/lib64

# Programs (每个 .cu 生成一个同名可执行文件)
TARGETS = gadi_bf16.exe gadi_fp32.exe gadi_fp64.exe

all: $(TARGETS)

# 通用规则：foo.exe 从 foo.cu 编译
%.exe: %.cu
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean