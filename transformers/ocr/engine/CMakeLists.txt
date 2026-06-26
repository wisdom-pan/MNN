cmake_minimum_required(VERSION 3.10)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
include_directories(${CMAKE_SOURCE_DIR}/tools/cv/include/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

if (MNN_SEP_BUILD)
    if (MNN_BUILD_SHARED_LIBS)
        # compile dynamic so, support Linux/Mac
        add_library(ocr SHARED ${SRCS})
        set_target_properties(ocr PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        target_link_libraries(ocr ${MNN_DEPS})
    else()
        add_library(ocr STATIC ${SRCS})
    endif()
    list(APPEND MNN_DEPS ocr)
else()
    add_library(ocr OBJECT ${SRCS})
endif()

add_executable(ocr_demo ${CMAKE_CURRENT_LIST_DIR}/ocr_demo.cpp)
target_link_libraries(ocr_demo ${MNN_DEPS})
