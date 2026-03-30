#include "vis.hpp"

Visualization::Visualization(
    std::shared_ptr<SharedStars> shared_stars_ptr,
    Camera camera,
    BloomPipeline bp,
    int width = 1280,
    int height = 720
){
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    m_window_width = width;
    m_window_height = height;

    GLFWwindow* window = glfwCreateWindow(width, height, "Star Vis", NULL, NULL);
    if (window == NULL)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glewExperimental = GL_TRUE;
    glewInit();
    glClearColor(0.f, 0.f, 0.f, 1.0f); // Clear with black

    

}

void Visualization::run(){
    
}

// Private Functions -----------------------

void Visualization::render_loop(){
    
}

void Visualization::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}  

void Visualization::process_input(GLFWwindow * window){
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
        glfwSetWindowShouldClose(window, true);
    }
}