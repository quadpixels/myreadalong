main: main.cpp
	g++ -g -O2 -std=c++20 $^ -I/usr/include/hdf5/serial -I/usr/include/opus \
		-L/usr/lib/x86_64-linux-gnu/hdf5/serial \
		-lhdf5_cpp -lhdf5 -lsndfile -lfftw3 -fopenmp -fopenmp-simd -lpthread -lgtest -lgtest_main -o $@

clean:
	@if [ -f main ] ; then rm -v main; fi
