### Standard settings ###
TARGET	= lens
SRC	= src/main.cpp src/math.cpp src/renderer.cpp src/screen_io.cpp src/lens.cpp
CXX	= g++
SHELL	= /bin/sh

USE_CCFITS = TRUE

### Check OpenCV version ###
ifneq ($(MAKECMDGOALS),clean)
ifeq ($(shell pkg-config opencv4 2>/dev/null; echo $$?), 0)
CV_NAME = opencv4
else ifeq ($(shell pkg-config opencv 2>/dev/null; echo $$?), 0)
CV_NAME = opencv
else
 $(error could not find or check opencv version)
endif
endif

CXXFLAGS = -O3 -std=c++11 -Wall -Wpedantic -flto $(shell pkg-config --cflags $(CV_NAME))
LIBS = $(shell pkg-config --libs $(CV_NAME)) -lstdc++ -lm
ifeq ($(USE_CCFITS), TRUE)
CCFITS_FLAGS = -lcfitsio -lCCfits
else
CCFITS_FLAGS = 
endif

all:
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LIBS) $(CCFITS_FLAGS) -D HAS_CCFITS=$(USE_CCFITS)
clean:
	rm -f $(TARGET)
