NVCC = nvcc
NVCCFLAGS = -Xcompiler -fPIC -shared -O3 -arch=sm_75
TARGET = libgemm.so
SRCS = src/main.cu src/cuda_naive.cu src/cuda_tiled.cu src/matrix_multiply.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
