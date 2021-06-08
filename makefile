CC=g++
CFLAGS=-O3 -Wall -Werror -Wextra -fopenmp -pthread
INCLUDE=/usr/local/cuda/include
LIB_PATH=/usr/local/cuda/lib64
LIB=cudart

TARGET=bwbench
OBJECTS=main.o gpu.o cpu.o
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(CC) -fopenmp $(OBJECTS) -o $@ -L$(LIB_PATH) -l$(LIB)

%.o: %.cc
	$(CC) $(CFLAGS) $< -c -I$(INCLUDE)

clean:
	rm -rf bwbench *.o
