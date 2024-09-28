// 2024-09-28

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <H5Cpp.h>

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