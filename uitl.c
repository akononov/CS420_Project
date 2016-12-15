#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>

void parse_args(int argc, char** argv, size_t* n, size_t* t) {
  int option = 0;

  // Default number of threads
  *t = 12;

  while ((option = getopt(argc, argv, "n:t:b")) != -1) {
    switch (option) {
      case 'n':
        sscanf(optarg, "%zu", n);
        break;
      case 't':
        sscanf(optarg, "%zu", t);
        break;
      case 'b':
      	sscanf(optargm "%zu", b);
      default:
        printf("Usage: %s -n NUM -t NUMTHREADS\n", argv[0]);
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
