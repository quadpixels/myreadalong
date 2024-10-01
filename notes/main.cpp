// 2024-09-28

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <assert.h>
#include <math.h>

#include <fftw3.h>
#include <H5Cpp.h>
#include <sndfile.h>

struct MyTensor;
MyTensor ComputeFBankData(const std::vector<short> wavsignal);

enum MyActivationFunc {
    LINEAR,
    RELU
};

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
    void Summary() {
        printf("rank=%zu (", dim.size());
        for (size_t i=0; i<dim.size(); i++) {
            if (i>0) { printf(","); }
            printf("%d", dim[i]);
        }
        printf(")\n");
    }
    MyTensor Reshape(const std::vector<int>& s) {
        int flatdim_out = std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<int>());
        assert(std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>()) == flatdim_out);
        MyTensor ret = *this;
        ret.dim = s;
        return ret;
    }
    int Rank() const { return int(dim.size()); }
};

struct MyConv2D {
    MyTensor kernel;
    MyTensor bias;
    MyActivationFunc activation = MyActivationFunc::LINEAR;
    MyConv2D(const MyTensor& k) : kernel(k) {
        assert(k.Rank() == 4);
        bias.InitZeroes({1});
    }
    MyConv2D(const MyTensor& k, const MyTensor& b) : kernel(k), bias(b) {}
    MyTensor operator()(MyTensor& in) {
        assert(in.Rank() == 4);
        std::vector<int> out_dim = in.dim;
        out_dim[3] = kernel.dim[3];
        std::vector<float> out_v(std::accumulate(out_dim.begin(), out_dim.end(), 1, std::multiplies<int>()));
        MyTensor ret(out_v, out_dim);

        // 1 channel input only for now
        if (in.dim[3] != 1) {
            assert(0 && "Unimplemented");
        } 

        for (int k=0; k<kernel.dim[3]; k++) {
            for (int y=0; y<out_dim[1]; y++) {
                for (int x=0; x<out_dim[2]; x++) {
                    float s = 0;
                    for (int dx = -kernel.dim[1]/2, kx=0; dx <= kernel.dim[1]/2; dx++, kx++) {
                        for (int dy = -kernel.dim[0]/2, ky=0; dy <= kernel.dim[0]/2; dy++, ky++) {
                            float in_elt = 0;
                            if (y+dy >= 0 && x+dx >= 0 && y+dy < in.dim[1] && x+dx < in.dim[2]) {
                                in_elt = in.at({0, y+dy, x+dx, 0});
                            }
                            s += kernel.at({ky, kx, 0, k}) * in_elt;
                        }
                    }
                    s += bias.at({k});
                    if (activation == MyActivationFunc::RELU) {
                        if (s < 0) s = 0;
                    }
                    ret.at({0, y, x, k}) = s;
                }
            }
        }
        return ret;
    }
};

struct MyBatchNormalization {
    float epsilon = 0.001f;
    MyTensor beta, gamma, moving_mean, moving_variance;
    MyBatchNormalization(const MyTensor& b, const MyTensor& g, const MyTensor& mm, const MyTensor& mv) :
        beta(b), gamma(g), moving_mean(mm), moving_variance(mv) {}
    MyBatchNormalization() {
        beta = MyTensor({0.0f}, {1});
        gamma = MyTensor({1.0f}, {1});
        moving_mean = MyTensor({0.0f}, {1});
        moving_variance = MyTensor({1.0f}, {1});
    }
    MyTensor operator()(MyTensor& in) {
        assert(in.Rank() == 4);
        MyTensor ret = in;
        for (int c=0; c<beta.dim[0]; c++) {
            const float g = gamma.at({c});
            const float b = beta.at({c});
            for (int y=0; y<in.dim[1]; y++) {
                for (int x=0; x<in.dim[2]; x++) {
                    ret.at({0,y,x,c}) = g * (in.at({0,y,x,c}) - moving_mean.at({c})) / sqrtf(moving_variance.at({c}) + epsilon) + b;
                }
            }
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
        return in.Reshape(outdim);
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
        H5::H5File file("onelayer.h5", H5F_ACC_RDONLY);
        // Step 3: Open the dataset
        MyTensor convkernel = ReadMyTensorFromH5(&file, "/conv2d_1/conv2d_1/kernel:0");
        MyTensor convbias   = ReadMyTensorFromH5(&file, "/conv2d_1/conv2d_1/bias:0");
        
        MyConv2D myconv2d(convkernel, convbias);
        MyTensor in0({7,2,3,3,8,
                      4,5,3,8,4,
                      3,3,2,8,4,
                      2,8,7,2,7,
                      5,4,4,5,4}, {1,5,5,1});
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
        t.Print();  // Expected to be equal to [[[ -8.346334  18.380442 -15.431074  -8.501173  18.683409]]]
        assert(fabs(t.at({0}) - ( -8.346334)) < 1e-4);
        assert(fabs(t.at({1}) - ( 18.380442)) < 1e-4);
        assert(fabs(t.at({2}) - (-15.431074)) < 1e-4);
        assert(fabs(t.at({3}) - ( -8.501173)) < 1e-4);
        assert(fabs(t.at({4}) - ( 18.683409)) < 1e-4);

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

MyTensor ComputeFBankData(const std::vector<short> wavsignal) {
  const int fs = 16000;
  const int time_window = 25;  // msec
  const int window_len = fs / 1000 * time_window;
  const int wav_length = int(wavsignal.size());
  const int range0_end = int(wav_length * 1.0f / fs * 1000 - time_window) / 10;
  printf("wav_length=%d, range0_end=%d\n", wav_length, range0_end);

  MyTensor ret({ range0_end, window_len/2 });

  for (int r=0; r<range0_end; r++) {
    int lb = r*160, ub = lb + window_len;
    double in[window_len];
    fftw_complex out[window_len];
    for (int i=0; i<window_len; i++) {
        if (lb+i < wav_length) {
            float w = 0.54f - 0.46f * cos(2 * 3.14159f * i / 399.0f);  // Hamming window
            in[i] = wavsignal[lb+i] * w;
        } else { in[i] = 0; }
    }
    fftw_plan p;
    p = fftw_plan_dft_r2c_1d(window_len, in, out, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i=0; i<window_len/2; i++) {
        ret.at({r, i}) = logf(sqrtf(out[i][0] * out[i][0] + out[i][1] * out[i][1]) + 1);
    }

    if (r==0 && false) {
        for (int i=0; i<window_len/2; i++) {
            printf("%g ", ret.at({r, i}));
            if (i%10 == 9) printf("\n");
        }
    }
    fftw_destroy_plan(p);
  }

  return ret;
}

MyTensor PadAndReshapeFBankData(MyTensor& fbanks) {
    const int fft_size = 200;
    assert(fbanks.dim.size() == 2);
    assert(fbanks.dim[1] == fft_size);
    const int outlen = ((fbanks.dim[0]-1) / 8 + 1) * 8;
    MyTensor ret({1, outlen, fft_size, 1});
    for (int y=0; y<fbanks.dim[0]; y++) {
        for (int x=0; x<fft_size; x++) {
            ret.at({0, y, x, 0}) = fbanks.at({y, x});
        }
    }
    return ret;
}

void SndfileTest() {
  // 1. Load sound data
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

  // 2. Compute FBank Data and pad+reshape FBank data
  MyTensor fbanks = ComputeFBankData(samples);
  fbanks = PadAndReshapeFBankData(fbanks);
  fbanks.Summary();
//   printf("%g\n", fbanks.at({0,0,0,0}));  // 9.5859
//   printf("%g\n", fbanks.at({0,0,1,0}));  // 9.20345
//   printf("%g\n", fbanks.at({0,1,0,0}));  // 8.97781

  // 3. Load weights and run it
  H5::H5File file("weights.h5", H5F_ACC_RDONLY);
  printf("Loading DFCNN params\n");
  printf("conv2d:\n");
  MyTensor conv2d_kernel = ReadMyTensorFromH5(&file, "/conv2d/conv2d/kernel:0");
  conv2d_kernel.Summary();
//   printf("%g\n", conv2d_kernel.at({0,0,0,0}));  // 0.310447
//   printf("%g\n", conv2d_kernel.at({0,0,0,1}));  // 0.292678
  MyTensor conv2d_bias = ReadMyTensorFromH5(&file, "/conv2d/conv2d/bias:0");
  conv2d_bias.Summary();
//   printf("%g\n", conv2d_bias.at({0}));  // 0.0140544
  MyConv2D conv2d(conv2d_kernel, conv2d_bias);
  conv2d.activation = MyActivationFunc::RELU;

  MyTensor blah = conv2d(fbanks);
  printf("conv2d output:\n");
  blah.Summary();
//   printf("%g\n", conv2d_out.at({0,0,0,0}));  // 0
//   printf("%g\n", conv2d_out.at({0,0,0,2}));  // 9.14927
//   printf("%g\n", conv2d_out.at({0,0,1,1}));  // 0.253471
//   printf("%g\n", conv2d_out.at({0,1,0,0}));  // 2.60429
//  printf("%g\n", conv2d_out.at({0,351,199,0}));  // 0.0140544

  printf("batch_normalization:\n");
  MyBatchNormalization batch_normalization(
    ReadMyTensorFromH5(&file, "/batch_normalization/batch_normalization/beta:0"),
    ReadMyTensorFromH5(&file, "/batch_normalization/batch_normalization/gamma:0"),
    ReadMyTensorFromH5(&file, "/batch_normalization/batch_normalization/moving_mean:0"),
    ReadMyTensorFromH5(&file, "/batch_normalization/batch_normalization/moving_variance:0")
  );
  blah = batch_normalization(blah);
  printf("batch_normalization output:\n");
  blah.Summary();
  printf("%g\n", blah.at({0,0,0,0}));

  file.close();
  sf_close(inFile);
}

int main() {
    H5open();
    MiniTest();
    SndfileTest();
    H5close();
    return 0;
}