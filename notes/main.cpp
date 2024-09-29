// 2024-09-28

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <H5Cpp.h>

struct MyTensor {
    std::vector<float> v;
    std::vector<int> dim;
    MyTensor(const std::vector<float>& _v, const std::vector<int>& _d) : v(_v), dim(_d) {}
    float& at(const std::vector<int>& idx) {
        int flatidx = 0;
        int mult = 1;
        for (int i=dim.size()-1; i>=0; i--) {
            int ii = idx.at(i);
            flatidx += ii * mult;
            mult *= dim.at(i);
        }
        return v.at(flatidx);
    }
    void Print() {
        std::vector<int> idx(dim.size());
        bool done = false;
        while (!done) {
            printf("%5g ", at(idx));
            idx.back()++;
            for (int i=idx.size()-1; i>=0; i--) {
                if (idx.at(i) >= dim.at(i)) {
                    if (i == 0) {
                        done = true;
                        break;
                    } else {
                        idx.at(i) = 0;
                        idx.at(i-1) ++;
                        if (i == 1) { printf("\n"); }
                    }
                }
            }
        }
    }
};

struct MyConv2D {
    MyTensor kernel;
    bool padding;
    MyConv2D(const MyTensor& k) : kernel(k) {}
    MyTensor operator()(MyTensor& in) {
        std::vector<int> out_dim = in.dim;
        std::vector<float> out_v(std::accumulate(in.dim.begin(), in.dim.end(), 1, std::multiplies<int>()));
        MyTensor ret(out_v, out_dim);
        for (int y=0; y<out_dim[0]; y++) {
            for (int x=0; x<out_dim[1]; x++) {
                float s = 0;
                for (int dx = -kernel.dim[1]/2, kx=0; dx <= kernel.dim[1]/2; dx++, kx++) {
                    for (int dy = -kernel.dim[0]/2, ky=0; dy <= kernel.dim[0]/2; dy++, ky++) {
                        float in_elt = 0;
                        if (y+dy >= 0 && x+dx >= 0 && y+dy < in.dim[0] && x+dx < in.dim[1]) {
                            in_elt = in.at({y+dy, x+dx});
                        }
                        s += kernel.at({ky, kx}) * in_elt;
                    }
                }
                ret.at({y, x}) = s;
            }
        }
        return ret;
    }
};

// Build:
// CFlags are obtained by `pkg-config --cflags hdf5-serial`
// LD Flags are inspired by `pkg-config --libs hdf5-serial`. One needs to add -lhdf5_cpp
//
// g++ main.cpp -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5

int main() {
    // Step 1: Initialize the HDF5 library (optional)
    H5open();
    try {
        // Step 2: Open the HDF5 file
        H5::H5File file("weights.h5", H5F_ACC_RDONLY);
        // Step 3: Open the dataset
        H5::DataSet dataset = file.openDataSet("/conv2d_1/conv2d_1/kernel:0");
        // Step 4: Get the data type and dimensions of the dataset
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        printf("rank=%d\n", rank);
        hsize_t dims[rank];
        dataspace.getSimpleExtentDims(dims, nullptr);
        printf("dims:");
        for (int i=0; i<rank; i++) { printf(" %llu", dims[i]); }  // (3,3,1,1)
        printf("\n");
        printf("type: %d\n", int(dataspace.getSimpleExtentType()));

        std::vector<float> data(std::accumulate(dims, dims+rank, 1, std::multiplies<int>()));

        // Step 5: Read the data into a vector
        dataset.read(data.data(), H5::PredType::NATIVE_FLOAT); // Read data into the vector. PredType means pre-defined type

        // Print the data
        for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < dims[1]; ++j) {
                std::cout << data[i * dims[1] + j] << " "; // Access the data
            }
            std::cout << std::endl;
        }

        MyConv2D myconv2d(
            MyTensor(data, {int(dims[0]), int(dims[1])})
        );
        
        MyTensor in0({7,2,3,3,8,
                      4,5,3,8,4,
                      3,3,2,8,4,
                      2,8,7,2,7,
                      5,4,4,5,4}, {5,5});
        MyTensor t = myconv2d(in0);
        t.Print();

        // Step 6: Close the dataset and file
        dataset.close();
        file.close();
    } catch (H5::FileIException &error) {
        std::cerr << "File I/O error: " << error.getCDetailMsg() << std::endl;
    } catch (H5::DataSetIException &error) {
        std::cerr << "Dataset I/O error: " << error.getCDetailMsg() << std::endl;
    } catch (H5::DataSpaceIException &error) {
        std::cerr << "DataSpace error: " << error.getCDetailMsg() << std::endl;
    } catch (H5::Exception &error) {
        std::cerr << "General error: " << error.getCDetailMsg() << std::endl;
    }

    // Step 7: Close the HDF5 library (optional)
    H5close();

    return 0;
}