// Exec pack: docs/development/m5_execution_pack.md T5.4
//
// Catch2 entry point wrapping MPI lifecycle for multi-rank comm tests.
// Individual TEST_CASE's stay MPI-agnostic and rely on rank/size queries
// via a helper in the test file itself.

#include <catch2/catch_session.hpp>

#include <mpi.h>

int main(int argc, char** argv) {
  // MPI_THREAD_SINGLE matches M5 scheduler assumption (D-M5-5). Catch2 is
  // single-threaded by default, so no upgrade is required.
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

  const int result = Catch::Session().run(argc, argv);

  // Drain any stragglers so MPI_Finalize doesn't complain about pending
  // messages (Iprobe-drop is acceptable since failures already fail the test).
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return result;
}
