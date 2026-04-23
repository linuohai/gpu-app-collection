GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM80 ?= -gencode=arch=compute_80,code=\"sm_80,compute_80\"

CPPSRC = ../graph_parser/util.cpp ../graph_parser/parse.cpp
CUSRC = mis.cu
EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/mis
OPTS = -O3
CUOPTS = $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM80)
INC = -I$(CUDA_INSTALL_PATH)/include
LIBS = -lcudart

OBJS = $(notdir $(CPPSRC:.cpp=.o))

all: $(OBJS)
	nvcc $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC) *.o $(INC) -o $(EXECUTABLE) $(LIBS)

$(OBJS):
	g++ -c $(CPPSRC) $(OPTS)

clean:
	rm -f *.o *.cu.cpp $(EXECUTABLE)

cleancucpp:
	rm -f *.cu.cpp
