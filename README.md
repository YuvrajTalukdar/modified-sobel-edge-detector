# modified-sobel-edge-detector
This software processes the sobel result and creates a binary image with only the relevent edges.
You need opencv installed to compile this code.
to compile the code type the below command in the terminal. Make sude the terminal current directory has the source code.
g++ -std=c++17 modified_sobel.cpp `pkg-config --cflags --libs opencv` -O2

To execute the program type
./a.out -src=boat.jpg

Replace boat.jpg with your image.
