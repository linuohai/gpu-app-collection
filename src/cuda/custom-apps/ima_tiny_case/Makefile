ifeq ($(GPUAPPS_SETUP_ENVIRONMENT_WAS_RUN), 0)
$(error You must run "source setup_environment" before calling make)
endif

EXECUTABLE := ima_tiny_case
CUFILES := ima_tiny_case.cu
CCFILES :=
OMIT_CUTIL_LIB := 1
NVCCFLAGS += -std=c++14

include ../../common/common.mk
