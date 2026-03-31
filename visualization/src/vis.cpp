#include "vis.hpp"

Visualization::Visualization(
    std::shared_ptr<SharedStars> shared_stars_ptr,
    Camera &camera,
    BloomPipeline &bp,
    ImguiUI &ui,
    float point_scale,
    int width,
    int height
){
    m_window_width = width;
    m_window_height = height;
    m_point_scale = point_scale;

    // Init OpenGL
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    GLFWwindow* m_window = glfwCreateWindow(width, height, "Star Vis", NULL, NULL);
    if (m_window == NULL)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    glewExperimental = GL_TRUE;
    glewInit();
    glClearColor(0.f, 0.f, 0.f, 1.0f); // Clear with black
    
    // Init Shaders
    m_point_sprite_shader = std::make_unique<Shader>(SHADER_DIR "point_sprites.vs", SHADER_DIR "point_sprites.fs");
    m_point_sprite_shader->setFloat("uPointScale", height * m_point_scale);

    // Init VAO VBO
    glGenVertexArrays(1, &m_stars_VAO);
    glGenBuffers(1, &m_stars_VBO);

    // Init projection_matrix
    m_projection_matrix = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.05f, 500.0f);

    // Don't need model matrix because stars are assumed to already be in our world space
    glm::mat4 vp_composite = m_projection_matrix * camera.get_view_matrix();
    m_point_sprite_shader->setMatrix4("mvp_composite", vp_composite);


    glEnable(GL_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // additive — stars glow through each other
    glDepthMask(GL_FALSE); // Prevent z-fighting at the far end of the scene

    // Init Bloom Pipeline
    bp.initialize_pipeline(width, height);

    // Init Imgui UI
    ui.initialize_imgui(m_window);
    
}

void Visualization::update_star_data(StarMapPtr stars){
    glBindVertexArray(m_stars_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_stars_VBO);
    
    // Copy vertices data into buffer's memory
    // glBufferData(GL_ARRAY_BUFFER, stars.size() * sizeof(StarVertex), stars.data(), GL_STATIC_DRAW);
    
    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)offsetof(StarVertex, position));
    // glEnableVertexAttribArray(0);
    
    // glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)offsetof(StarVertex, magnitude));
    // glEnableVertexAttribArray(1);
    
    // glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)offsetof(StarVertex, color));
    // glEnableVertexAttribArray(2);
}

void Visualization::run(){
    
}

// Private Functions -----------------------

void Visualization::render_loop(){
    
    while(!glfwWindowShouldClose(m_window)){



    }

    std::cout << "Terminating Visualization" << std::endl;
    
    glfwTerminate();
}

void Visualization::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}  

void Visualization::process_input(){
    if(glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
        glfwSetWindowShouldClose(m_window, true);
    }
}