// Exec pack: docs/development/m8_execution_pack.md T8.0 (T7.8b carry-forward)
//
// MPI-aware Catch2 entry point for the 2-rank GPU overlap test. Mirrors
// tests/{runtime,comm,scheduler}/main_mpi.cpp — each rank initialises MPI
// before any TEST_CASE starts and finalises cleanly afterwards.

#include <catch2/catch_session.hpp>

#include <mpi.h>

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

  const int result = Catch::Session().run(argc, argv);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return result;
}
