main: main.cpp
	g++ -g -std=c++20 $^ -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5 -o $@

clean:
	@if [ -f main ] ; then rm -v main; fi
