


```
nvcc cf.cpp or cf.cu && ./a.out

clear && nvcc cf.cpp -I ./include -o cf.out && ./cf.out
clear && nvcc cf.cu -I ./include -o cf.cu && ./cf.out

clear && nvcc -I ./include -o ./bin/radknl radial_search_cuda.cpp radial_search.cu  && ./bin/radknl
```