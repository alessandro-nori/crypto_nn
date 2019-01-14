IDIR = include
ODIR = build
SDIR = src
LDIR = lib
BDIR = bin

TARGET = bin/crypto_nn

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

$(TARGET): $(OBJ)
	@mkdir -p $(BDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDIR)/fhe.a $(LDLIBS)

$(ODIR)/main.o: $(SDIR)/main.cpp
	@mkdir -p $(ODIR)
	@$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: $(SDIR)/%.cpp $(IDIR)/%.h
	@mkdir -p $(ODIR)
	@$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	@rm -rf $(ODIR) $(TARGET) 
