# Configure testing.
set(TEST_BINS gradient_descent.test)
file(GLOB_RECURSE TEST_SOURCE_FILES ${PROJECT_ROOT_DIR}/tests/*.c)
message(STATUS "${TEST_SOURCE_FILES}")

include(AddCatch2)

message(STATUS "Testing enabled. Building test binaries ...")
add_executable(${TEST_BINS} ${TEST_SOURCE_FILES})

target_include_directories(${TEST_BINS} PUBLIC ${PROJECT_ROOT_DIR}/src)
target_include_directories(${TEST_BINS} PUBLIC ${PROJECT_ROOT_DIR}/third_party/Catch2/include)
target_include_directories(${TEST_BINS} PRIVATE ${MATLIBR_INCLUDE_DIRS} ${OPENBLAS_INCLUDE_DIRS})
target_link_libraries(${TEST_BINS} PRIVATE 
    ${PROJECT_LIBRARY_NAME}
    ${MATLIBR_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(${TEST_BINS})
