cmake_minimum_required(VERSION 3.27.2)

project(gradient_descent)

set(CMAKE_C_COMPILER g++)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})

set(PROJECT_LIBRARY_NAME gradient_descent)
set(PROJECT_EXECUTABLE_NAME run)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

add_executable(${PROJECT_EXECUTABLE_NAME} main.c ${SOURCE_FILES})
add_library(${PROJECT_LIBRARY_NAME} STATIC ${SOURCE_FILES})

find_package(openblas REQUIRED)
find_package(matlibr REQUIRED)

target_include_directories(${PROJECT_EXECUTABLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_EXECUTABLE_NAME} PRIVATE ${OPENBLAS_INCLUDE_DIRS} ${MATLIBR_INCLUDE_DIRS})
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE m dl ${OPENBLAS_LIBRARIES} ${MATLIBR_LIBRARIES})

target_include_directories(${PROJECT_LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_EXECUTABLE_NAME} PRIVATE ${OPENBLAS_INCLUDE_DIRS} ${MATLIBR_INCLUDE_DIRS})
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE m dl ${OPENBLAS_LIBRARIES} ${MATLIBR_LIBRARIES})

enable_testing()

add_subdirectory(tests)
