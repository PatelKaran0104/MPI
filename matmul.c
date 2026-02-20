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

void fillLocalRows(double *arr, int row_start, int num_rows, int n, int seed, int value)
{
    for (int i = 0; i < num_rows; i++) {
        int global_i = row_start + i;
        for (int j = 0; j < n; j++) {
            unsigned long state = concatenate(global_i, j) + seed + value;
            arr[i * n + j] = my_rand(&state, 0, 1);
        }
    }
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

    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));

    int offset = 0;
    for (int r = 0; r < size; r++) {
        int r_rows = base_rows + (r < remainder ? 1 : 0);
        recvcounts[r] = r_rows * n;
        displs[r]     = offset;
        offset += r_rows * n;
    }

    int my_rows  = base_rows + (rank < remainder ? 1 : 0);
    int my_start = rank * base_rows + (rank < remainder ? rank : remainder);

    /* ---- Allocate memory ---- */
    double *A_local = (double *)malloc((size_t)my_rows * n * sizeof(double));
    double *B       = (double *)malloc((size_t)n * n * sizeof(double));
    double *B_local = (double *)malloc((size_t)my_rows * n * sizeof(double));
    double *C_local = (double *)calloc((size_t)my_rows * n, sizeof(double));
    double *C       = NULL;

    if (rank == 0)
        C = (double *)malloc((size_t)n * n * sizeof(double));

    /* ---- Each process initializes its own rows locally ---- */
    fillLocalRows(A_local, my_start, my_rows, n, seed, 0);
    fillLocalRows(B_local, my_start, my_rows, n, seed, 1);

    /* ---- Reconstruct full B on all processes ----*/
    MPI_Allgatherv(B_local, my_rows * n, MPI_DOUBLE,
                   B, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(B_local);

    /* ---- Matrix multiplication: C_local = A_local x B ---- */
    for (int i = 0; i < my_rows; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A_local[i * n + k];
            for (int j = 0; j < n; j++) {
                C_local[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }

    /* ---- Gather result rows at rank 0 ---- */
    MPI_Gatherv(C_local, my_rows * n, MPI_DOUBLE,
                C, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* ---- Output (rank 0 only) ---- */
    if (rank == 0) {
        if (verbose == 1 && n <= 10) {
            double *full_A = (double *)malloc((size_t)n * n * sizeof(double));
            fillMatrix(full_A, n, seed, 0);
            printMatrix("Matrix A", full_A, n);
            free(full_A);

            printMatrix("Matrix B", B, n);
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
    free(B);
    free(C_local);
    free(recvcounts);
    free(displs);
    if (rank == 0)
        free(C);

    MPI_Finalize();
    return 0;
}
