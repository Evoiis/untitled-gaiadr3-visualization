#include "bloom_pipeline.hpp"

BloomPipeline::BloomPipeline(int width, int height){

    unsigned int m_hdrFBO, m_color_buffer;
    unsigned int m_brightnessFBO, m_brightness_buffer;

    // HDR FBO Setup ----------------------
    glGenFramebuffers(1, &m_hdrFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdrFBO);

    // Color texture attachment
    glGenTextures(1, &m_color_buffer);
    glBindTexture(GL_TEXTURE_2D, m_color_buffer);
    // Allocate texture memory
    // GL_RGBA16F allows values > 1.0 for over-bright stars
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    // Use lerp to sample the texture when scaling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Clamp edges
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Attach the texture to the FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_color_buffer, 0);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
        std::cerr << "FBO not complete" << std::endl;
    }        
    
    // Bright FBO Setup -------------------------------------
    
    glGenFramebuffers(1, &m_brightnessFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_brightnessFBO);

    glGenTextures(1, &m_brightness_buffer);
    glBindTexture(GL_TEXTURE_2D, m_brightness_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Clamp edges
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_brightness_buffer, 0);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Bright FBO not complete" << std::endl;


    // Init Quad VAO VBO
    glGenVertexArrays(1, &m_quad_VAO);
    glGenBuffers(1, &m_quad_VBO);
    glBindVertexArray(m_quad_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_quad_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(m_quad_vertices), m_quad_vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Blur Ping Pong FBO
    glGenFramebuffers(2, m_blurFBO);
    glGenTextures(2, m_blur_buffer);
    for(int i = 0; i < 2; i++){
        glBindFramebuffer(GL_FRAMEBUFFER, m_blurFBO[i]);
        glBindTexture(GL_TEXTURE_2D, m_blur_buffer[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // Clamp edges
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_blur_buffer[i], 0);
    }







    // Cleanup
    glBindFramebuffer(GL_FRAMEBUFFER, 0);



}
