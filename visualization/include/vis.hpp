#ifndef VIS_H
#define VIS_H

#include <GL/glew.h> 
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>

#include "star_data.hpp"
#include "camera.hpp"
#include "bloom_pipeline.hpp"
#include "imgui_ui.hpp"

#include <iostream>

class Visualization{
public:

    Visualization(
        std::shared_ptr<SharedStars> shared_stars_ptr,
        Camera camera,
        BloomPipeline bp,
        int width,
        int height
    );

    void run();

private:
    int m_window_width;
    int m_window_height;
    GLFWwindow* window;

    void render_loop();
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    void process_input(GLFWwindow* window);

};

#endif
