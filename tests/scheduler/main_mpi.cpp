// Exec pack: docs/development/m5_execution_pack.md T5.7
//
// MPI-aware Catch2 entry point for 2-rank scheduler integration tests.
// Mirrors tests/comm/main_mpi.cpp — we ship our own main so every rank
// initialises MPI before any TEST_CASE starts and finalises cleanly
// afterwards.

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
