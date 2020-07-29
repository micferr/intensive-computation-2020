#ifndef PBR_H
#define PBR_H

#include <memory>
#include <thread>
#include <map>
#include <vector>

template<typename T>
struct block_base {
    virtual std::vector<T> multiply_by_vector(const std::vector<T>& v, size_t offset=0) const = 0;
};

template<
    typename T,
    T N0, T N1, T N2, T N3,
    T N4, T N5, T N6, T N7,
    T N8, T N9, T N10, T N11,
    T N12, T N13, T N14, T N15
> struct block : public block_base<T> {
    std::vector<T> multiply_by_vector(const std::vector<T>& v, size_t offset=0) const {
        std::vector<T> y(4,0);

        if constexpr(N0) { y[0] += v[0+offset]*N0; }
        if constexpr(N1) { y[0] += v[1+offset]*N1; }
        if constexpr(N2) { y[0] += v[2+offset]*N2; }
        if constexpr(N3) { y[0] += v[3+offset]*N3; }

        if constexpr(N4) { y[1] += v[0+offset]*N4; }
        if constexpr(N5) { y[1] += v[1+offset]*N5; }
        if constexpr(N6) { y[1] += v[2+offset]*N6; }
        if constexpr(N7) { y[1] += v[3+offset]*N7; }

        if constexpr(N8) { y[2] += v[0+offset]*N8; }
        if constexpr(N9) { y[2] += v[1+offset]*N9; }
        if constexpr(N10) { y[2] += v[2+offset]*N10; }
        if constexpr(N11) { y[2] += v[3+offset]*N11; }

        if constexpr(N12) { y[3] += v[0+offset]*N12; }
        if constexpr(N13) { y[3] += v[1+offset]*N13; }
        if constexpr(N14) { y[3] += v[2+offset]*N14; }
        if constexpr(N15) { y[3] += v[3+offset]*N15; }

        return y;
    }

    static constexpr int count_nnz() {
        int nnz = 0;

        if constexpr (N0) {nnz++;}
        if constexpr (N1) {nnz++;}
        if constexpr (N2) {nnz++;}
        if constexpr (N3) {nnz++;}

        if constexpr (N4) {nnz++;}
        if constexpr (N5) {nnz++;}
        if constexpr (N6) {nnz++;}
        if constexpr (N7) {nnz++;}

        if constexpr (N8) {nnz++;}
        if constexpr (N9) {nnz++;}
        if constexpr (N10) {nnz++;}
        if constexpr (N11) {nnz++;}

        if constexpr (N12) {nnz++;}
        if constexpr (N13) {nnz++;}
        if constexpr (N14) {nnz++;}
        if constexpr (N15) {nnz++;}

        return nnz;
    }
};

template<typename T, int Rows>
struct pbr {
    std::map<std::pair<unsigned,unsigned>,std::shared_ptr<block_base<T>>> blocks;
    std::vector<unsigned> rem_cols;
    std::vector<unsigned> rem_rows;
    std::vector<T> rem_val;

    std::vector<T> multiply_by_vector(const std::vector<T>& v) {
        std::vector<T> y(Rows, 0);
        std::vector<std::pair<unsigned,std::vector<T>>> partial_y(blocks.size());
        std::vector<std::thread> threads;

        auto kernel = [&](const std::pair<std::pair<unsigned,unsigned>,std::shared_ptr<block_base<T>>>& coord_block, unsigned block_idx){
            const auto& coords = coord_block.first;
            const auto i = coords.first;
            const auto j = coords.second;
            const auto& curr_block = coord_block.second.get();

            std::vector<T> partial_product = curr_block->multiply_by_vector(v, 4*j);
            partial_y[block_idx] = {i*4, partial_product};
        };

        // Perform partial products in parallel
        unsigned block_idx = 0;
        for (const auto& coord_block : blocks) {
            threads.emplace_back(std::move(std::thread(kernel, coord_block, block_idx)));
            block_idx++;

        }
        for (auto& t : threads) {
            t.join();
        }

        // Sum partial results (can be map-reduced by block row)
        for (const auto& py : partial_y) {
            unsigned offset = py.first;
            const auto& vec = py.second;
            for (unsigned i = 0; i < vec.size(); i++) {
                y[i + offset] += vec[i];
            }
        }

        // Handle excess elements
        for (unsigned i = 0; i < rem_val.size(); i++) {
            unsigned col = rem_cols[i];
            unsigned row = rem_rows[i];
            T val = rem_val[i];

            y[row] += val*v[col];
        }

        return y;
    }
};

#endif // PBR_H
