// Exec pack: docs/development/m5_execution_pack.md T5.8
//
// MPI-aware Catch2 entry point for the 2-rank runtime tests. Mirrors
// tests/comm/main_mpi.cpp and tests/scheduler/main_mpi.cpp — we ship our own
// main so every rank initialises MPI before any TEST_CASE starts and finalises
// cleanly afterwards.

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
