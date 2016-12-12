for omp_schedule in static dynamic dynamic,2 dynamic,4 dynamic,8
        do

        dir=OMP_$omp_schedule
        cp -r inputs $dir
        cd $dir

        sed "s/SCHEDULE/$omp_schedule/" spmv_openmp_temp > spmv_openmp.c
        make clean
        make

        qsub run_openmp.sub

        cd ..

        done
