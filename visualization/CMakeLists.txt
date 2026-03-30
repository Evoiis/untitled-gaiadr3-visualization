cmake_minimum_required(VERSION 3.10)
project(GaiaDR3Vis)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)


protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
    ../protobuf_msgs/star_data.proto
)


set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/submodules/imgui)

add_executable(main
    src/main.cpp
    # src/vis.cpp
    src/shader.cpp
    src/node.cpp

    src/camera.cpp
    src/bloom_pipeline.cpp
    src/imgui_ui.cpp

    # Protobuf
    ${PROTO_SRCS}
    ${PROTO_HDRS}

    # ImGui core
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp

    # ImGui backends
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_compile_definitions(main PRIVATE
    SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/"
)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${Protobuf_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(main
    OpenGL::GL
    glfw
    GLEW::GLEW
    ${Protobuf_LIBRARIES}
    ${ZMQ_LIBRARIES}
)
