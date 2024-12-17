# Makefile for compiling main.cpp and running job-submit.sh

# Compiler
CXX = g++
CXXFLAGS = -fopenmp -O2 -Wall

# Targets
TARGET = main
SOURCES = main.cpp
SCRIPT = job-submit.sh

# Default target
all: $(TARGET)

# Compile the main.cpp
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Run the job-submit.sh script
run-script: $(SCRIPT)
	./$(SCRIPT)

# Clean up
clean:
	rm -f $(TARGET)

.PHONY: all run-script clean