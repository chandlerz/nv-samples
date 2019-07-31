ECHO = @echo
AT =

CUDA_PATH =/usr/local/cuda-10.1
OUTNAME = gemm_imma
CC   := g++
CUCC := $(CUDA_PATH)/bin/nvcc

CFLAGS  := -m64 -std=c++11 
CUFLAGS := -m64 -std=c++11 

# if debug
# CFLAGS  += -DDEBUG_
# CUFLAGS += -DDEBUG_

# if use customized kernel for transformation
CFLAGS  += -DCUSTOMERIZED_KERNEL 
CUFLAGS += -DCUSTOMERIZED_KERNEL 

# CFLAGS  += -g -lineinfo
# CUFLAGS += -g -G

# Gencode arguments
SMS ?= 75 # only sm_75 or above supports IMMA

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_COM_PATH = $(CUDA_PATH)/samples/common/inc
INCLUDES := -I$(CUDA_COM_PATH) -I$(CUDA_INC_PATH) -I/usr/include
CFLAGS += $(INCLUDES)
CUFLAGS += $(INCLUDES)

CUDA_LIB_PATH = $(CUDA_PATH)/lib64
LDPATH := -L$(CUDA_LIB_PATH) -L/usr/lib -L/usr/lib/x86_64-linux-gnu

LDFLAGS :=  $(LDPATH) -lcudart -lcublas -lcublasLt
LDFLAGS += -Wl,-rpath=$(CUDA_LIB_PATH)

SRCDIR := ./
OBJDIR := ../../obj
BINDIR := ../../bin

OBJS   = $(patsubst %.cpp, $(OBJDIR)/%.o, $(notdir $(wildcard $(SRCDIR)/*.cpp)))
CUOBJS = $(patsubst %.cu,  $(OBJDIR)/%.o, $(notdir $(wildcard $(SRCDIR)/*.cu)))

all: build

build: $(BINDIR)/$(OUTNAME)

$(BINDIR)/$(OUTNAME): $(OBJS) $(CUOBJS)
	$(AT)if [ ! -d $(BINDIR) ]; then mkdir -p $(BINDIR); fi
	$(ECHO) Linking: $@
	$(AT)$(CC) -o $@ $^ $(LDFLAGS) 

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CUCC) $(CUFLAGS) $(GENCODE_FLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR)/* $(BINDIR)/$(OUTNAME)
