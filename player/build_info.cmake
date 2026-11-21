# Run at every build (cmake -P): writes OUT with the build number, "build <commits> (<short hash>)", plus
# "-modified" when tracked files have uncommitted changes. Only rewrites OUT when the text changes, so an
# unchanged build number doesn't recompile anything. Without git (a source zip): "build unknown".
execute_process(COMMAND git rev-list --count HEAD WORKING_DIRECTORY "${SRC}" OUTPUT_VARIABLE count
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE r1)
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY "${SRC}" OUTPUT_VARIABLE hash
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE r2)
execute_process(COMMAND git status --porcelain --untracked-files=no WORKING_DIRECTORY "${SRC}"
                OUTPUT_VARIABLE dirty OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(r1 EQUAL 0 AND r2 EQUAL 0 AND count MATCHES "^[0-9]+$" AND hash MATCHES "^[0-9a-f]+$")
  set(build "build ${count} (${hash}")
  if(dirty)
    string(APPEND build "-modified")
  endif()
  string(APPEND build ")")
else()
  set(build "build unknown")
endif()
set(text "#pragma once\n#define OPEN_PLAYER_BUILD \"${build}\"\n")
if(EXISTS "${OUT}")
  file(READ "${OUT}" old)
endif()
if(NOT old STREQUAL text)
  file(WRITE "${OUT}" "${text}")
endif()
