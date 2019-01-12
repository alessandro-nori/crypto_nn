IDIR = include
ODIR = obj
SDIR = src

CC = g++
CFLAGS = -g -O2 -std=c++11 -pthread -DFHE_THREADS -DFHE_BOOT_THREADS -fmax-errors=2

LD = g++
AR = ar
ARFLAGS=rv
GMP=-lgmp
NTL=-lntl

LDLIBS = -L/usr/local/lib $(NTL) $(GMP) -lm

_DEPS = layer.h relu.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o layer.o relu.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/main.o: $(SDIR)/main.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/layer.o: $(SDIR)/layer.cpp $(IDIR)/layer.h
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/relu.o: $(SDIR)/relu.cpp $(IDIR)/relu.h
	$(CC) -c -o $@ $< $(CFLAGS)

crypto_nn: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ HElib/fhe.a $(LDLIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o
