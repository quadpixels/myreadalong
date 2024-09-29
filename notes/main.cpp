// 2024-09-28

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <assert.h>
#include <math.h>

#include <H5Cpp.h>
#include <sndfile.h>

struct MyTensor {
    std::vector<float> v;
    std::vector<int> dim;
    MyTensor(const std::vector<float>& _v, const std::vector<int>& _d) : v(_v), dim(_d) {}
    MyTensor(const std::vector<int>& d) : dim(d) {
        v.resize(std::accumulate(d.begin(), d.end(), 1, std::multiplies<int>()));
    }
    MyTensor() {}
    void InitZeroes(const std::vector<int>& d) {
        dim = d;
        v.clear();
        v.resize(std::accumulate(d.begin(), d.end(), 1, std::multiplies<int>()));
    }
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
        if (dim.size() == 1) printf("\n");
    }
};

struct MyConv2D {
    int num_kernels;
    MyTensor kernel;
    MyTensor bias;
    bool padding;
    MyConv2D(const MyTensor& k) : kernel(k), num_kernels(1) { bias.InitZeroes({1}); }
    MyConv2D(const MyTensor& k, const MyTensor& b) : kernel(k), bias(b), num_kernels(1) {}
    MyTensor operator()(MyTensor& in) {
        std::vector<int> out_dim = in.dim;
        std::vector<float> out_v(std::accumulate(in.dim.begin(), in.dim.end(), 1, std::multiplies<int>()));
        MyTensor ret(out_v, out_dim);

        for (int k=0; k<num_kernels; k++) {
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
                    s += bias.at({k});
                    if (num_kernels == 1) {
                        ret.at({y, x}) = s;
                    } else {
                        assert(0 && "Not implemented");
                    }
                }
            }
        }
        return ret;
    }
};

struct MyBatchNormalization {
    float epsilon = 0.001f;
    float moving_variance = 1;
    float moving_mean = 0;
    MyTensor operator()(MyTensor& in) {
        MyTensor ret = in;
        for (size_t i=0; i<in.v.size(); i++) {
            ret.v[i] = (in.v[i] - moving_mean) / sqrtf(moving_variance + epsilon);
        }
        return ret;
    }
};

struct MyDense {
    MyTensor kernel;
    MyTensor bias;
    MyDense(const MyTensor& k, const MyTensor& b) : kernel(k), bias(b) {}
    MyTensor operator()(MyTensor& in) {
        assert(in.dim.size() == 1);
        assert(in.dim[0] == kernel.dim[0]);
        MyTensor out({kernel.dim[1]});
        
        for (int k=0; k<kernel.dim[1]; k++) {
            float elt = 0;
            for (int y=0; y<in.dim[0]; y++) {
                elt += in.at({y}) * kernel.at({y, k});
            }
            elt += bias.at({k});
            out.at({k}) = elt;
        }
        
        return out;
    }
};

struct MyReshape {
    std::vector<int> outdim;
    MyReshape(const std::vector<int>& o) : outdim(o) {}
    MyTensor operator()(MyTensor& in) {
        int flatdim_in = std::accumulate(in.dim.begin(), in.dim.end(), 1, std::multiplies<int>());
        int flatdim_out = std::accumulate(outdim.begin(), outdim.end(), 1, std::multiplies<int>());
        assert(flatdim_in = flatdim_out);
        MyTensor out = in;
        out.dim = outdim;
        return out;
    }
};

// Build:
// CFlags are obtained by `pkg-config --cflags hdf5-serial`
// LD Flags are inspired by `pkg-config --libs hdf5-serial`. One needs to add -lhdf5_cpp
//
// g++ main.cpp -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5

MyTensor ReadMyTensorFromH5(const H5::H5File* file, const char* path, int force_rank = -1) {
    H5::DataSet dataset = file->openDataSet(path);
    H5::DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    hsize_t dims[rank];
    dataspace.getSimpleExtentDims(dims, nullptr);
    if (force_rank != -1) {
        for (int i=force_rank; i<rank; i++) {
            assert(dims[i] == 1);
        }
        rank = force_rank;
    }

    std::vector<float> data(std::accumulate(dims, dims + rank, 1, std::multiplies<int>()));
    dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);
    std::vector<int> d(dims, dims+rank);
    dataset.close();
    return MyTensor(data, d);
}

void MiniTest() {
    try {
        // Step 2: Open the HDF5 file
        H5::H5File file("weights.h5", H5F_ACC_RDONLY);
        // Step 3: Open the dataset
        MyTensor convkernel = ReadMyTensorFromH5(&file, "/conv2d_1/conv2d_1/kernel:0", 2);
        MyTensor convbias   = ReadMyTensorFromH5(&file, "/conv2d_1/conv2d_1/bias:0", 1);
        
        MyConv2D myconv2d(convkernel, convbias);
        MyTensor in0({7,2,3,3,8,
                      4,5,3,8,4,
                      3,3,2,8,4,
                      2,8,7,2,7,
                      5,4,4,5,4}, {5,5});
        MyTensor t = myconv2d(in0);
        t.Print();

        MyBatchNormalization mybn;
        t = mybn(t);
        t.Print();

        MyTensor densekernel = ReadMyTensorFromH5(&file, "/dense_1/dense_1/kernel:0", 2);
        MyTensor densebias   = ReadMyTensorFromH5(&file, "/dense_1/dense_1/bias:0", 1);

        MyReshape myreshape({25});
        t = myreshape(t);
        t.Print();

        MyDense mydense(densekernel, densebias);
        t = mydense(t);
        t.Print();

        // Step 6: Close the dataset and file
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
}

void SndfileTest() {
  const char *fn;
  SNDFILE *inFile;
  SF_INFO sfinfo;

  fn = "laihongduiquyan.wav";

  inFile = sf_open(fn, SFM_READ, &sfinfo);

  printf("Sample Rate=%d Hz\n", sfinfo.samplerate);
  printf("Channels=%d\n", sfinfo.channels);
  printf("Format=%d ", sfinfo.format);
  if ((sfinfo.format & SF_FORMAT_WAV) == SF_FORMAT_WAV) {
    printf("(WAV) ");
  }
  if ((sfinfo.format & SF_FORMAT_PCM_16) == SF_FORMAT_PCM_16) {
    printf("(signed 16bit) ");
  }
  printf("\n");
  printf("Frames=%ld\n", sfinfo.frames);
  assert(sfinfo.channels == 1);

  std::vector<short> samples(sfinfo.frames);
  sf_readf_short(inFile, samples.data(), sfinfo.frames);
  printf("[");
  for (int i=0; i<3; i++) {
    printf("%hd ", samples[i]);
  }
  printf(" ...");
  for (int i=sfinfo.frames-3; i<sfinfo.frames; i++) {
    printf(" %hd", samples[i]);
  }
  printf("]\n");

  sf_close(inFile);
}

int main() {
    H5open();
    MiniTest();
    SndfileTest();
    H5close();
    return 0;
}