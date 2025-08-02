# 最低 CMake 版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(STITCH_BAOGAI)

# 设置 C++ 标准（核心补充）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++17)  # 强制编译器使用 C++17（关键补充）

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV 找到，版本: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "未找到 OpenCV，请检查安装路径")
endif()

# 收集 src 目录下的所有源文件
file(GLOB_RECURSE SOURCES 
    "src/*.c"
    "src/*.cpp"
    "src/*.h"
    "src/*.hpp"
)

# 指定可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

# 生成可执行文件
add_executable(main ${SOURCES})

# 链接 OpenCV 库
target_link_libraries(main ${OpenCV_LIBS})

# 指定头文件目录
target_include_directories(main PUBLIC ${PROJECT_SOURCE_DIR}/src)