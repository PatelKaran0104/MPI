CC = mpicc
CFLAGS = -O3 -Wall

all: matmul matrix matmul_v2

matmul: matmul.c
	$(CC) $(CFLAGS) -o matmul matmul.c -lm

matmul_v2: matmul_v2.c
	$(CC) $(CFLAGS) -o matmul_v2 matmul_v2.c -lm

matrix: matrix.c
	$(CC) $(CFLAGS) -o matrix matrix.c -lm

clean:
	rm -f matmul matmul_v2 matrix

.PHONY: all clean