# RunF2pyQuiet.cmake  --  run an f2py command with stdout silenced.
#
# Inputs (via -D on the cmake -P command line):
#   F2PY_CMD   – semicolon list of the full command + arguments
#   WORKDIR    – working directory for execute_process

execute_process(
  COMMAND ${F2PY_CMD}
  WORKING_DIRECTORY "${WORKDIR}"
  OUTPUT_QUIET
  ERROR_QUIET
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  # Re-run without suppression so the error is visible.
  execute_process(
    COMMAND ${F2PY_CMD}
    WORKING_DIRECTORY "${WORKDIR}"
  )
  message(FATAL_ERROR "f2py command failed (exit ${_rc})")
endif()
