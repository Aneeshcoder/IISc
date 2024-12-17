# Makefile for MPI code

# Compiler
MPICXX = mpicxx

# Flags
CXXFLAGS = -O2 -Wall

# Target executable
TARGET = main

# Source files
SRCS = main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Rule to link the target executable
$(TARGET): $(OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ $^

# Rule to compile source files
%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Rule to submit the job
submit: $(TARGET)
	bash job-submit.sh

.PHONY: all clean submit