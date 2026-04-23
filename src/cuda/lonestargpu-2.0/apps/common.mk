BIN    		:= $(TOPLEVEL)/bin
INPUTS 		:= $(TOPLEVEL)/inputs

NVCC 		:= nvcc
GCC  		:= g++
CC := $(GCC)
#CUB_DIR := $(TOPLEVEL)/../cub

# Compiler-specific flags — SM50+ only (SM10-SM35 removed for CUDA 12.x)
GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM80 ?= -gencode=arch=compute_80,code=\"sm_80,compute_80\"

ifdef debug
FLAGS := $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM80) -g -DLSGDEBUG=1 -G
else
# including -lineinfo -G causes launches to fail because of lack of resources, pity.
FLAGS := -O3 $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM80) -g -Xptxas -v  #-lineinfo -G
endif
INCLUDES := -I $(TOPLEVEL)/include -I $(NVIDIA_COMPUTE_SDK_LOCATION)/common/inc
LINKS := 

EXTRA := $(FLAGS) $(NVCC_ADDITIONAL_ARGS) $(INCLUDES) $(LINKS)

.PHONY: clean variants support optional-variants

ifdef APP
$(APP): $(SRC) $(INC)
	$(NVCC) $(EXTRA) -DVARIANT=0 -o $@ $<
	cp $@ $(BIN)

variants: $(VARIANTS)

optional-variants: $(OPTIONAL_VARIANTS)

support: $(SUPPORT)

clean: 
	rm -f $(APP) $(BIN)/$(APP)
ifdef VARIANTS
	rm -f $(VARIANTS)
endif
ifdef OPTIONAL_VARIANTS
	rm -f $(OPTIONAL_VARIANTS)
endif

endif
