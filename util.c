#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>

#define CACHE_SIZE 12582912

void parse_args(int argc, char** argv, int* n, int* d, int* b, int* t) {
  int option = 0;

  // Default number of threads
  *t = 32;

  while ((option = getopt(argc, argv, "n:d:b:t:")) != -1) {
    switch (option) {
      case 'n':
        sscanf(optarg, "%zu", n);
        printf("Got parameter n...%d\n",*n);
        break;
      case 'd':
        sscanf(optarg, "%zu", d);
        printf("Got parameter d...%d\n",*d);
        break;
      case 'b':
      	sscanf(optarg, "%zu", b);
      	printf("Got parameter b...%d\n",*b);
      	break;
      case 't':
      	sscanf(optarg, "%zu", t);
      	printf("Got parameter t...%d\n",*t);
		break;
      default:
        printf("Usage: %s -n NUM -d NUMTHREADS -b BLOCKSIZE -t TILESIZE\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }
}

void clear_cache() {
  size_t array_size = CACHE_SIZE * 2.5;
  double temp = 0.0;
  double* dummy = (double*)malloc(sizeof(double) * array_size);

  for (size_t i = 0; i < array_size; ++i) {
    dummy[i] = i / array_size;
  }

  for (size_t j = 0; j < 10; ++j) {
    for (size_t i = 0; i < array_size; ++i) {
      temp += dummy[i] / j;
    }
  }

  fprintf(stderr, "Cleared cache (ignore this value: %f)\n", temp);
}
