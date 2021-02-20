cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)

#
# PROJECT
#

project(benchmark CUDA CXX)

#
# STATIC VARIABLES
#

# Define Release by default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "CMAKE_BUILD_TYPE not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # EXPORT COMPILE COMMANDS
set(CMAKE_VERBOSE_MAKEFILE ON)

#
# OUTPUTS
#

set(OUTPUT_DEBUG Debug/bin)
set(OUTPUT_RELEASE Release/bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE
    "${CMAKE_BINARY_DIR}/${OUTPUT_RELEASE}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE
    "${CMAKE_BINARY_DIR}/${OUTPUT_RELEASE}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE
    "${CMAKE_BINARY_DIR}/${OUTPUT_RELEASE}")

#
# COMPILER/LINKER FLAGS
#

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-DNDEBUG)
endif()


if(NOT EXISTS ${CMAKE_CXX_COMPILER})
  set(CMAKE_CXX_COMPILER "clang++")
  message(STATUS "CMAKE_CXX_COMPILER is not specified: Use clang++ by default.")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native")
set(
  CMAKE_CXX_FLAGS
  "-Wall -Wextra -Wcast-align -Wcast-qual -Wdelete-non-virtual-dtor -Weffc++ -Wold-style-cast -Woverloaded-virtual -Wpointer-arith -Wfloat-equal -Winit-self -Wredundant-decls -Wconversion -Wshadow -Wsign-promo -Wswitch-default -Wswitch-enum -Wvariadic-macros -Wwrite-strings -Wno-unknown-pragmas")

#
# EXPORT COMPILE COMMANDS
#
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(${CMAKE_EXPORT_COMPILE_COMMANDS} STREQUAL ON)
  add_custom_target(gen_compile_commands ALL
                    COMMENT "Generating compile_commands.json")
  add_custom_command(TARGET gen_compile_commands PRE_BUILD
                     COMMAND ${CMAKE_COMMAND}
                             -E
                             remove
                             ${CMAKE_SOURCE_DIR}/compile_commands.json)
  find_program(SED sed)
  if(SED)
    add_custom_command(TARGET gen_compile_commands PRE_BUILD
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                       COMMAND sed
                               -i
                               -e
                               's/nvcc/clang++/g'
                               -e
                               's/-x cuda/-x cu/g'
                               -e
                               's/-x cu/-x cuda/g'
                               ${CMAKE_BINARY_DIR}/compile_commands.json
                       COMMENT "Replacing strings")
  endif()
  find_program(COMPDB compdb)
  if(COMPDB)
    add_custom_command(TARGET gen_compile_commands PRE_BUILD
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                       COMMAND ${COMPDB}
                               list
                               >
                               ${CMAKE_SOURCE_DIR}/compile_commands.json
                       COMMENT "Running compdb")
  else()
    message(AUTHOR_WARNING "'compdb' not found")
    add_custom_command(TARGET gen_compile_commands PRE_BUILD
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                       COMMAND ${CMAKE_COMMAND}
                               -E
                               create_symlink
                               ${CMAKE_BINARY_DIR}/compile_commands.json
                               ${CMAKE_SOURCE_DIR}/compile_commands.json
                       COMMENT "Creating symbolic link")
  endif()
endif()

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "found OpenMP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(AUTHOR_WARNING "OpenMP not found")
endif()

#
# SUBDIRECTORIES
#

# main project
add_subdirectory(src)