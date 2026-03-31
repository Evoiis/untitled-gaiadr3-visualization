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
#include "shader.hpp"

#include <iostream>

class Visualization{
public:

    Visualization(
        std::shared_ptr<SharedStars> shared_stars_ptr,
        Camera &camera,
        BloomPipeline &bp,
        ImguiUI &ui,
        float point_scale = 2.0f,
        int width = 1280,
        int height = 720
    );

    void run();

    void update_star_data(StarMapPtr stars);

private:
    int m_window_width;
    int m_window_height;
    unsigned int m_stars_VAO;
    unsigned int m_stars_VBO;
    GLFWwindow* m_window;

    float m_point_scale;

    std::unique_ptr<Shader> m_point_sprite_shader;

    glm::mat4 m_projection_matrix;
    glm::mat4 m_vp_matrix;
    

    void render_loop();
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    void process_input();

};

#endif
