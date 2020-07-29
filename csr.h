#ifndef CSR_H
#define CSR_H

#include <thread>
#include <vector>

#include "utils.h"

/**
 * Struct to represent sparse matrices in CSR format
 */
template<typename T>
struct csr {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<T> val;
    int cols;

    csr(const std::vector<T>& matrix, int n, int m) : row_ptr(n+1, 0), cols(m) {
        int nnz = 0;
        for (int i = 0; i < n; i++) {
            int row_nnz = 0;
            for (int j = 0; j < m; j++) {
                T v = matrix[i*m+j];
                if (v != 0) {
                    row_nnz++;
                    col_idx.emplace_back(j);
                    val.emplace_back(v);
                }
            }
            nnz += row_nnz;
            row_ptr[i+1] = nnz;
        }
    }

    std::vector<T> extract_row(int row) {
        std::vector<T> res(cols, 0);
        for (int i = row_ptr[row]; i < row_ptr[row+1]; i++) {
            res[col_idx[i]] = val[i];
        }
        return res;
    }

    int get_numrows() {
        return row_ptr.size()-1;
    }

    int get_numcols() {
        return cols;
    }

    std::vector<T> multiply_by_vector(const std::vector<T>& v) {
        int num_rows = get_numrows();
        std::vector<T> result(num_rows, 0);
        for (int i = 0; i < num_rows; i++) {
            const std::vector<T> row = extract_row(i);
            for (int j = 0; j < cols; j++) {
                result[i] += row[j]*v[j];
            }
        }
        return result;
    }

    std::vector<T> multiply_by_vector_parallel(const std::vector<T>& v) {
        int num_rows = get_numrows();
        std::vector<T> result(num_rows, 0);

        // Kernel to apply in parallel, computes product for one row
        auto kernel = [&v, &result, this](int index){
            std::vector<T> row = extract_row(index);
            for (int j = 0; j < cols; j++) {
                result[index] += row[j]*v[j];
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(num_rows);
        for (int i = 0; i < num_rows; i++) {
            std::thread thread(kernel, i);
            threads.emplace_back(std::move(thread));
        }

        for (int i = 0; i < num_rows; i++) {
            threads[i].join();
        }
        return result;
    }

    void print() {
        print_vector(row_ptr);
        print_vector(col_idx);
        print_vector(val);
    }
};

#endif // CSR_H
