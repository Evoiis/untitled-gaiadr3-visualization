cmake_minimum_required(VERSION 3.10)
project(LearnOpenGL)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/submodules/imgui)

add_executable(main
    src/main.cpp

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
)

target_link_libraries(main
    OpenGL::GL
    glfw
    GLEW::GLEW
)
