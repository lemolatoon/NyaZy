# Locate all the .cpp files in the test directory
file(GLOB_RECURSE TEST_FILES *.cpp)

# Create an executable for running tests from the test files
add_executable(test_nyacc ${TEST_FILES})

# Link with necessary libraries (e.g., LLVM, if needed)
target_link_libraries(test_nyacc ${LLVM_LIBS})
