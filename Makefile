CC = g++

INCDIR=include
SRCDIR=src
OBJDIR=obj
LIBDIR=lib

vpath %.cpp $(SRCDIR)
vpath %.o $(OBJDIR)

CFLAGS=-Wall -Werror -pedantic -std=c++11  -g -O2

_DEPS=Const.h Network.h Layer.h ActivationFunction.h Trainer.h LossFunction.h
DEPS= nnetwork.h $(patsubst %,$(INCDIR)/%,$(_DEPS)) 

_OBJ=ActivationFunction.o Layer.o  Network.o LossFunction.o Trainer.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ)) 

$(LIBDIR)/libnnetwork.a: $(OBJ)
	ar -cvq $@ $(OBJ)
	#$(CC) -o $@ -shared -static-libstdc++ $(OBJ)

test: test.cpp nnetwork.h $(LIBDIR)/libnnetwork.a
	$(CC) -o $@ $<  $(CFLAGS) -Llib -lnnetwork

$(OBJ): $(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $<   $(CFLAGS)

.PHONY: clean
clean:
	-rm $(OBJ) $(LIBDIR)/libnnetwork.a test

