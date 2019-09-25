CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

#=========================================================
# Project

SET(PROJECT_NAME CustomOp)
PROJECT(${PROJECT_NAME} CXX CUDA)

#=========================================================
# Misc

SET(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

SET( CMAKE_EXE_LINKER_FLAGS  "-ltensorflow_framework" )
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET (BUILD_SHARED_LIBS TRUE)

SET(projectDir "")
STRING(REPLACE "/cmake"
       "" projectDir
       ${CMAKE_SOURCE_DIR})

#=========================================================
# Includes and Libs

#base path include
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../)

#tensorflow include
INCLUDE_DIRECTORIES("/path/to/python3.6/site-packages/tensorflow/include/")

#tensorflow lib
SET(tensorflowLib "/path/to/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so")

#=========================================================
# Files to compile
MESSAGE(${tensorflowLib}, "TESTING: PRINTING TFLIB")
FILE(GLOB_RECURSE cppFiles
       ${CMAKE_SOURCE_DIR}/../src/filter/*.cpp
       ${CMAKE_SOURCE_DIR}/../src/filter/*.h
)

FILE(GLOB cudaFiles
       ${CMAKE_SOURCE_DIR}/../src/filter/*.cu
       ${CMAKE_SOURCE_DIR}/../src/utils/*.cu
       ${CMAKE_SOURCE_DIR}/../src/utils/*.h
)

#=========================================================
# Output

ADD_DEFINITIONS(-DGOOGLE_CUDA -D_USE_MATH_DEFINES)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${projectDir}/binaries")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${projectDir}/binaries")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${projectDir}/binaries")

ADD_LIBRARY(${PROJECT_NAME} SHARED ${cppFiles} ${cudaFiles})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${tensorflowLib} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${tensorflowLib})
