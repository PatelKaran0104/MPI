#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

double my_rand(unsigned long *state, double lower, double upper)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    unsigned long x = (*state * 0x2545F4914F6CDD1DULL);
    const double inv = 1.0 / (double)(1ULL << 53);
    double u = (double)(x >> 11) * inv;
    return lower + (upper - lower) * u;
}

unsigned concatenate(unsigned x, unsigned y)
{
    unsigned pow = 10;
    while (y >= pow)
        pow *= 10;
    return x * pow + y;
}

void fillMatrix(double *arr, int n, int seed, int value)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            unsigned long state = concatenate(i, j) + seed + value;
            arr[i * n + j] = my_rand(&state, 0, 1);
        }
    }
}

void printMatrix(const char *label, double *arr, int n)
{
    printf("%s:\n", label);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > 0) printf(" ");
            printf("%f", arr[i * n + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n       = (argc > 1) ? atoi(argv[1]) : 8000;
    int seed    = (argc > 2) ? atoi(argv[2]) : 42;
    int verbose = (argc > 3) ? atoi(argv[3]) : 0;

    double t_start = MPI_Wtime();

    /* ---- Row distribution ---- */
    int base_rows = n / size;
    int remainder = n % size;

    int *row_counts  = (int *)malloc(size * sizeof(int));
    int *row_offsets = (int *)malloc(size * sizeof(int));
    int *sendcounts  = (int *)malloc(size * sizeof(int));
    int *displs      = (int *)malloc(size * sizeof(int));

    int max_block_rows = base_rows + (remainder > 0 ? 1 : 0);
    int offset = 0;
    for (int r = 0; r < size; r++) {
        row_counts[r]  = base_rows + (r < remainder ? 1 : 0);
        row_offsets[r]  = offset;
        sendcounts[r] = row_counts[r] * n;
        displs[r]     = offset * n;
        offset += row_counts[r];
    }

    int my_rows = row_counts[rank];

    /* ---- Allocate memory ---- */
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    double *A_local = (double *)malloc((size_t)my_rows * n * sizeof(double));
    double *C_local = (double *)calloc((size_t)my_rows * n, sizeof(double));

    double *B_block = (double *)malloc((size_t)max_block_rows * n * sizeof(double));
    double *B_recv  = (double *)malloc((size_t)max_block_rows * n * sizeof(double));

    /* ---- Rank 0 initializes full A and B ---- */
    if (rank == 0) {
        A = (double *)malloc((size_t)n * n * sizeof(double));
        B = (double *)malloc((size_t)n * n * sizeof(double));
        C = (double *)malloc((size_t)n * n * sizeof(double));
        fillMatrix(A, n, seed, 0);
        fillMatrix(B, n, seed, 1);
    }

    /* ---- Distribute rows of A ---- */
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 A_local, my_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* ---- Distribute row-blocks of B ---- */
    MPI_Scatterv(B, sendcounts, displs, MPI_DOUBLE,
                 B_block, my_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(B);
        B = NULL;
    }

    /* ---- Ring-based multiplication with non-blocking overlap ---- */
    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    int current_source = rank;

    for (int step = 0; step < size; step++) {
        int src_rows   = row_counts[current_source];
        int src_offset = row_offsets[current_source];

        MPI_Request requests[2];
        int next_source = (current_source + 1) % size;

        /* Start non-blocking ring shift */
        if (step < size - 1) {
            int send_count = src_rows * n;
            int recv_count = row_counts[next_source] * n;

            MPI_Isend(B_block, send_count, MPI_DOUBLE, left, step,
                      MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(B_recv, recv_count, MPI_DOUBLE, right, step,
                      MPI_COMM_WORLD, &requests[1]);
        }

        /* Compute partial C while communication happens in background */
        for (int i = 0; i < my_rows; i++) {
            for (int k = 0; k < src_rows; k++) {
                double a_ik = A_local[i * n + (src_offset + k)];
                for (int j = 0; j < n; j++) {
                    C_local[i * n + j] += a_ik * B_block[k * n + j];
                }
            }
        }

        /* Wait for transfer to complete */
        if (step < size - 1) {
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

            double *tmp = B_block;
            B_block = B_recv;
            B_recv  = tmp;

            current_source = next_source;
        }
    }

    /* ---- Gather result rows at rank 0 ---- */
    MPI_Gatherv(C_local, my_rows * n, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* ---- Output (rank 0 only) ---- */
    if (rank == 0) {
        if (verbose == 1 && n <= 10) {
            printMatrix("Matrix A", A, n);

            double *B_print = (double *)malloc((size_t)n * n * sizeof(double));
            fillMatrix(B_print, n, seed, 1);
            printMatrix("Matrix B", B_print, n);
            free(B_print);

            printMatrix("Matrix C (Result)", C, n);
        }

        double checksum = 0.0;
        for (int i = 0; i < n * n; i++)
            checksum += C[i];
        printf("Checksum: %f\n", checksum);
    }

    double t_end = MPI_Wtime();

    if (rank == 0)
        printf("Execution time with %d ranks: %.2f s\n", size, t_end - t_start);

    /* ---- Cleanup ---- */
    free(A_local);
    free(C_local);
    free(B_block);
    free(B_recv);
    free(row_counts);
    free(row_offsets);
    free(sendcounts);
    free(displs);
    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
