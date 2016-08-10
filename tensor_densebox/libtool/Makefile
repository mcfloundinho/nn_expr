all: nms.so

libnms.a: nms.cpp nms.h
	g++-4.9 -o nms.o -c nms.cpp --std=c++11 -fPIC
	ar cr libnms.a nms.o

nms.so: nms_impl.pyx libnms.a setup.py
	python setup.py build_ext --inplace
	make clean

clean:
	rm -rf nms.o libnms.a build nms_impl.cpp

purge:
	make clean
	rm -f nms.so nms.cpython-35m-x86_64-linux-gnu.so
