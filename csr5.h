#ifndef CSR5_H
#define CSR5_H

#include <algorithm>
#include <atomic>

template<typename T>
struct csr5 {
    struct tile {
        std::vector<int> y_offset;
        std::vector<int> seg_offset;
        std::vector<int> empty_offset;
        std::vector<bool> bit_flag;

        tile(int omega, int sigma) : y_offset(omega, 0), seg_offset(omega, 0), empty_offset(0,0), bit_flag(omega * sigma, false) {};

        void print() const {
            std::cout << "Begin Tile:\n";
            std::cout << "Y Offset: ";
            print_vector(y_offset);
            std::cout << "Seg Offset: ";
            print_vector(seg_offset);
            std::cout << "Empty offset: ";
            print_vector(empty_offset);
            std::cout << "Bit Flag: ";
            print_vector(bit_flag);
            std::cout << "End Tile\n";
        }
    };

    unsigned omega, sigma;
    std::vector<int> row_ptr;
    std::vector<int> tile_ptr;
    std::vector<int> col_idx;
    std::vector<T> val;
    std::vector<tile> tile_desc;

    csr5(const std::vector<T>& matrix, int n, int m, unsigned _omega, unsigned _sigma) : omega(_omega), sigma(_sigma) {
        // Compute matrix indo
        int nnz = std::count_if(matrix.begin(), matrix.end(), [](const T& element){return element != 0;});
        int num_tiles = nnz/(omega*sigma);

        // Allocate memory for support structures
        row_ptr.resize(n+1, 0);
        tile_ptr.resize(num_tiles + 1 + (nnz%(omega*sigma) ? 1 : 0));
        col_idx.resize(nnz);
        val.resize(nnz);
        tile_desc.resize(num_tiles, tile(omega, sigma));

        row_ptr.back() = nnz;
        tile_ptr.back() = n;

        // Fill data structures
        int nnz_count = 0;
        int tile_count = 0;
        int tile_elements = omega*sigma;
        int last_row = -1;
        for (unsigned i = 0; i < matrix.size(); i++) {
            T element = matrix[i];
            if (element) {
                int curr_row = i/m;
                int curr_col = i%n;
                if (tile_count < num_tiles) {
                    bool is_new_row = curr_row != last_row;
                    int tile_idx = linear_to_tile_index(nnz_count);

                    tile_desc[tile_count].bit_flag[tile_idx] = is_new_row || !nnz_count;
                    col_idx[tile_count*tile_elements + tile_idx] = curr_col;
                    val[tile_count*tile_elements + tile_idx] = element;
                    if (is_new_row) {
                        for (int j = last_row+1; j <= curr_row; j++) {
                            row_ptr[j] = tile_count*tile_elements + nnz_count;
                        }
                        last_row = curr_row;
                    }

                    nnz_count++;
                    if (nnz_count == tile_elements) {
                        nnz_count = 0;
                        tile_count++;
                        tile_ptr[tile_count] = curr_row; // Current row
                    }
                } else { // Excess elements
                    int idx = num_tiles*tile_elements + nnz_count;
                    val[idx] = element;
                    col_idx[idx] = curr_col;
                    nnz_count++;
                }
            }
        }

        // This section can be easily parallelized

        // Generate y_offset, seg_offset, empty_offser
        for (unsigned tile_count = 0; tile_count < tile_desc.size(); tile_count++) {
            auto& t = tile_desc[tile_count];
            std::vector<bool> tmp_bit(omega, false);
            for (unsigned j = 1; j < omega; j++) {
                int acc = 0;
                for (unsigned i = 0; i < sigma; i++) {
                    acc += t.bit_flag[i*omega + j-1];
                    tmp_bit[j] = t.bit_flag[i*omega + j] ? true : tmp_bit[j];
                }
                t.y_offset[j] = acc;
                t.y_offset[j] += t.y_offset[j-1];
            }

            for (int j = omega-2; j >= 0; j--) {
                t.seg_offset[j] = (t.seg_offset[j+1]+1)*!tmp_bit[j+1];
            }

            auto bit_flags = reduction_sum(t.bit_flag, sigma, omega);
            auto contains_empty_row = false;
            for (int i = 0; i < bit_flags; i++) {
                if (row_ptr[tile_ptr[tile_count] + i] == row_ptr[tile_ptr[tile_count] + i + 1]) {
                    contains_empty_row = true;
                    break;
                }
            }
            if (contains_empty_row) {
                t.empty_offset.resize(bit_flags, 0);
                auto offset_counter = 0;
                auto curr_empty_bit = 0;
                for (int i = 0; i < bit_flags; i++) {
                    if (row_ptr[tile_ptr[tile_count] + i] == row_ptr[tile_ptr[tile_count] + i + 1]) {
                        bit_flags++;
                    } else {
                        t.empty_offset[curr_empty_bit++] = offset_counter;
                    }
                    offset_counter++;
                }
            }
        }
    }

    std::vector<T> multiply_by_vector(const std::vector<T>& v) {
        unsigned num_rows = row_ptr.size()-1;
        std::vector<std::vector<T>> Y(tile_desc.size());

        for (unsigned tile_idx = 0; tile_idx < tile_desc.size(); tile_idx ++) {
            std::vector<T> y(num_rows, 0);
            std::vector<T> tmp(omega, 0);
            std::vector<T> last_tmp(omega, 0);
            auto& curr_tile = tile_desc[tile_idx];
            auto& y_offset = curr_tile.y_offset;
            auto& seg_offset = curr_tile.seg_offset;
            auto& empty_offset = curr_tile.empty_offset;
            auto& bit_flag = curr_tile.bit_flag;
            auto contains_empty_rows = empty_offset.size() > 0;

            for (unsigned i = 0; i < omega; i++) {
                T sum = 0;
                bool is_red = !bit_flag[i];

                for (unsigned j = 0; j < sigma; j++) {
                    int ptr = tile_idx*omega*sigma + j*omega + i;
                    sum = sum + val[ptr] * v[col_idx[ptr]];
                    if (j != sigma-1 && bit_flag[(j+1)*omega + i]) {
                        if (is_red) { // end of a red segment
                            tmp[i-1] = sum;
                            is_red = false;
                        } else { // end of a green segment
                            if (!contains_empty_rows) {
                                y[tile_ptr[tile_idx] + y_offset[i]] = sum;
                                y_offset[i] = y_offset[i] + 1;
                            } else {
                                y[tile_ptr[tile_idx] +  empty_offset[y_offset[i]]] = sum;
                                empty_offset[y_offset[i]] =  empty_offset[y_offset[i]] + 1;
                            }
                        }
                        sum = 0;
                    }
                }
                if (is_red) { // end of a red segment
                    tmp[i-1] = sum;
                    sum = 0; // nullifies next instruction
                }
                last_tmp[i] = sum; //end of a blue sub-segment
            }
            fast_segmented_sum(tmp, seg_offset);
            for (unsigned i = 0; i < omega; i++) {
                if (!contains_empty_rows) {
                    last_tmp[i] = last_tmp[i] + tmp[i];
                    y[tile_ptr[tile_idx] + y_offset[i]] = last_tmp[i];
                } else {
                    last_tmp[i] = last_tmp[i] + tmp[i];
                    y[tile_ptr[tile_idx] + empty_offset[y_offset[i]]] = last_tmp[i];
                }
            }
            Y[tile_idx] = y;
            print_vector(y);
        }

        std::vector<T> result(num_rows, 0);
        for (const auto& y : Y) {
            for (unsigned i = 0; i < y.size(); i++) {
                result[i] += y[i];
            }
        }

        // handle excess elements
        for (unsigned i = tile_desc.size()*omega*sigma; i < val.size(); i++) {
            T el = val[i] * v[col_idx[i]];
            result[col_idx[i]] += el;
        }

        return result;
    }

    int coordinates_to_tile_index(int i, int j) {
        return i*sigma + j;
    }

    int linear_to_tile_index(int idx) {
        int i = idx%sigma;
        int j = idx/sigma;
        return coordinates_to_tile_index(i,j);
    }

    void print() const {
        std::cout << "Row Ptr: ";
        print_vector(row_ptr);
        std::cout << "Col Idx: ";
        print_vector(col_idx);
        std::cout << "Val: ";
        print_vector(val);
        std::cout << "Tile Ptr: ";
        print_vector(tile_ptr);
        for (const auto& t : tile_desc) {
            t.print();
        }
    }
};

#endif // CSR5_H
