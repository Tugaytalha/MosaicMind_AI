g++ -std=c++17 -o main main.cpp \
    -I /home/alierdem/libtorch/include \
    -I /home/alierdem/libtorch/include/torch/csrc/api/include \
    -L /home/alierdem/libtorch/lib \
        -Wl,-rpath,/home/alierdem/libtorch/lib \
    -ltorch -lc10 -ltorch_cpu -ljsoncpp 2> err.txt
