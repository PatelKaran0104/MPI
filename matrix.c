// SUMMA: Scalable Universal Matrix Multiply Algorithm
// Reference: van de Geijn, R. A., & Watts, J. (1997).
// https://doi.org/10.1002/(SICI)1096-9128(199704)9:4<255::AID-CPE250>3.0.CO;2-2


// Program structure (gather, verify):
// github.com/irifed/mpi101
#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Reference: Professor's provided random number generator from PDF
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

// Reference: Professor's provided concatenate function from PDF
unsigned concatenate(unsigned x, unsigned y)
{
    unsigned pow = 10;
    while (y >= pow)
        pow *= 10;
    return x * pow + y;
}

// Find greatest common divisor (used for panel width in SUMMA)
int greatest_common_divisor(int row, int col)
{
    while (col != 0)
    {
        int remainder = row % col;
        row = col;
        col = remainder;
    }
    return row;
}

// Find max near square grid dimensions that divide n (2D Process grid)
void find_grid_dimensions(int total_processes, int matrix_size, int *grid_rows, int *grid_cols)
{
    for (int possible_processes = total_processes; possible_processes >= 1; possible_processes--)
    {
        int max_possible_rows = (int)sqrt((double)possible_processes);

        for (int possible_rows = max_possible_rows; possible_rows >= 1; possible_rows--)
        {
            if (possible_processes % possible_rows != 0)
                continue;

            int possible_cols = possible_processes / possible_rows;

            if (matrix_size % possible_rows == 0 &&
                matrix_size % possible_cols == 0)
            {
                *grid_rows = possible_rows;
                *grid_cols = possible_cols;
                return;
            }
        }
    }

    // Fallback if nothing valid found
    *grid_rows = 1;
    *grid_cols = 1;
}

// Fill local block with values based on global indices and seed
void fill_local_block(double *local_block, int block_row, int block_col, int process_row, int process_col, uint64_t seed)
{
    int global_row_offset = process_row * block_row;
    int global_col_offset = process_col * block_col;

    for (int row = 0; row < block_row; row++)
    {
        for (int col = 0; col < block_col; col++)
        {
            int global_row = global_row_offset + row;
            int global_col = global_col_offset + col;

            unsigned long state = concatenate(global_row, global_col) + seed;

            local_block[row * block_col + col] = my_rand(&state, 0.0, 1.0);
        }
    }
}

// Perform local matrix multiplication for given block sizes
void local_multiply(const double *A, const double *B, double *C, int result_rows, int result_cols, int common_dim)
{
    for (int row = 0; row < result_rows; row++)
        for (int k = 0; k < common_dim; k++)
        {
            double a = A[row * common_dim + k];
            for (int col = 0; col < result_cols; col++)
                C[row * result_cols + col] += a * B[k * result_cols + col];
        }
}

// Print matrix when verbose is 1
void print_matrix(const double *matrix, int size)
{
    for (int row = 0; row < size; row++)
    {
        for (int col = 0; col < size; col++)
            printf("%.6f ", matrix[row * size + col]);
        printf("\n");
    }
    printf("\n");
}

// Gather local blocks from all processes to the root process and assemble the global matrix
void gather_blocks_to_root(double *local_block, double *global_matrix, int global_matrix_size,
                           int block_rows, int block_cols, int grid_rows, int grid_cols,
                           int rank, MPI_Comm comm)
{
    if (rank == 0)
    {
        // copy root's own block
        for (int row = 0; row < block_rows; row++)
            for (int col = 0; col < block_cols; col++)
                global_matrix[row * global_matrix_size + col] = local_block[row * block_cols + col];

        // receive blocks from others processes
        for (int src_rank = 1; src_rank < grid_rows * grid_cols; src_rank++)
        {
            double *buf = malloc(block_rows * block_cols * sizeof(double));
            MPI_Recv(buf, block_rows * block_cols, MPI_DOUBLE, src_rank, 0, comm, MPI_STATUS_IGNORE);

            int process_row = src_rank / grid_cols;
            int process_col = src_rank % grid_cols;
            for (int i = 0; i < block_rows; i++)
                for (int j = 0; j < block_cols; j++)
                    global_matrix[(process_row * block_rows + i) * global_matrix_size + (process_col * block_cols + j)] = buf[i * block_cols + j];
            free(buf);
        }
    }
    else
        MPI_Send(local_block, block_rows * block_cols, MPI_DOUBLE, 0, 0, comm);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4)
    {
        if (rank == 0)
            printf("Usage: %s <n> <seed> <verbose>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    uint64_t seed = strtoull(argv[2], NULL, 10);
    int verbose = atoi(argv[3]);

    double start_time = MPI_Wtime();

    int grid_rows, grid_cols;
    find_grid_dimensions(size, n, &grid_rows, &grid_cols);
    int grid_size = grid_rows * grid_cols;

    MPI_Comm active_comm;
    int color = (rank < grid_size) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &active_comm);

    if (active_comm == MPI_COMM_NULL)
    {
        double end_time = MPI_Wtime();
        MPI_Finalize();
        return 0;
    }

    int active_rank;
    MPI_Comm_rank(active_comm, &active_rank);

    int my_row = active_rank / grid_cols;
    int my_col = active_rank % grid_cols;

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(active_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(active_comm, my_col, my_row, &col_comm);

    int block_rows = n / grid_rows;
    int block_cols = n / grid_cols;

    double *A = calloc(block_rows * block_cols, sizeof(double));
    double *B = calloc(block_rows * block_cols, sizeof(double));
    double *C = calloc(block_rows * block_cols, sizeof(double));

    int panel_width = greatest_common_divisor(block_rows, block_cols);
    int num_panel_steps = n / panel_width;

    double *A_panel = calloc(block_rows * panel_width, sizeof(double));
    double *B_panel = calloc(panel_width * block_cols, sizeof(double));

    fill_local_block(A, block_rows, block_cols, my_row, my_col, seed);
    fill_local_block(B, block_rows, block_cols, my_row, my_col, seed + 1);

    for (int k = 0; k < num_panel_steps; k++)
    {
        int a_owner = (k * panel_width) / block_cols;
        int a_offset = (k * panel_width) % block_cols;

        if (my_col == a_owner)
            for (int i = 0; i < block_rows; i++)
                for (int j = 0; j < panel_width; j++)
                    A_panel[i * panel_width + j] = A[i * block_cols + a_offset + j];

        MPI_Bcast(A_panel, block_rows * panel_width, MPI_DOUBLE, a_owner, row_comm);

        int b_owner = (k * panel_width) / block_rows;
        int b_offset = (k * panel_width) % block_rows;

        if (my_row == b_owner)
            for (int i = 0; i < panel_width; i++)
                for (int j = 0; j < block_cols; j++)
                    B_panel[i * block_cols + j] = B[(b_offset + i) * block_cols + j];

        MPI_Bcast(B_panel, panel_width * block_cols, MPI_DOUBLE, b_owner, col_comm);

        local_multiply(A_panel, B_panel, C, block_rows, block_cols, panel_width);
    }

    double local_sum = 0.0;
    for (int i = 0; i < block_rows * block_cols; i++)
        local_sum += C[i];

    double checksum = 0.0;
    MPI_Reduce(&local_sum, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, active_comm);

    if (verbose && n <= 10)
    {
        double *fullA = NULL, *fullB = NULL, *fullC = NULL;

        if (active_rank == 0)
        {
            fullA = calloc(n * n, sizeof(double));
            fullB = calloc(n * n, sizeof(double));
            fullC = calloc(n * n, sizeof(double));
        }

        gather_blocks_to_root(A, fullA, n, block_rows, block_cols, grid_rows, grid_cols, active_rank, active_comm);

        gather_blocks_to_root(B, fullB, n, block_rows, block_cols, grid_rows, grid_cols, active_rank, active_comm);

        gather_blocks_to_root(C, fullC, n, block_rows, block_cols, grid_rows, grid_cols, active_rank, active_comm);

        if (active_rank == 0)
        {
            printf("Matrix A:\n");
            print_matrix(fullA, n);

            printf("Matrix B:\n");
            print_matrix(fullB, n);

            printf("Matrix C (Result):\n");
            print_matrix(fullC, n);

            free(fullA);
            free(fullB);
            free(fullC);
        }
    }

    if (active_rank == 0)
        printf("Checksum: %.6f\n", checksum);

    MPI_Barrier(active_comm);
    double end_time = MPI_Wtime();

    if (active_rank == 0)
        printf("Execution time with %d ranks: %.2f s\n", size, end_time - start_time);

    free(A);
    free(B);
    free(C);
    free(A_panel);
    free(B_panel);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&active_comm);

    MPI_Finalize();
    return 0;
}