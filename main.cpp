#include <iostream>

#include "csr.h"
#include "csr5.h"
#include "pbr.h"

using namespace std;

int main()
{
    vector<int> M = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    csr<int> c(M, 4, 4);
    print_vector(c.multiply_by_vector({1,1,1,1}));
    print_vector(c.multiply_by_vector_parallel({1,1,1,1}));

    vector<int> M2 = {
        1,0,2,3,0,0,4,5,
        0,1,0,2,0,0,0,0,
        0,0,0,0,0,0,0,0,
        1,2,3,4,5,0,6,7,
        0,1,0,2,0,3,0,0,
        1,2,0,0,0,0,0,0,
        0,1,2,3,4,5,6,7,
        1,2,3,4,5,6,7,8,
    };
    csr5<int> c2(M2, 8, 8, 4, 4);
    c2.print();
    print_vector(c2.multiply_by_vector({1,1,1,1,1,1,1,1}));

    std::cout << "PBR\n";
    block<int,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15> b;
    print_vector(b.multiply_by_vector({1,1,1,1}));

    pbr<int, 10> p;
    p.blocks[{1,1}] = std::make_shared<block<int,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15>>(b);
    print_vector(p.multiply_by_vector({1,1,1,1,1,1,1,1,1,1}));

    return 0;
}
