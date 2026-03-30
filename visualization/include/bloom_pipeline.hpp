#ifndef BPL_H
#define BPL_H

#include <GL/glew.h> 
#include <GLFW/glfw3.h>
#include <iostream>
#include <array>

class BloomPipeline{
public:

    BloomPipeline(int width, int height);

    void run();

private:
    unsigned int m_hdrFBO;
    unsigned int m_color_buffer;
    unsigned int m_brightFBO;
    unsigned int m_bright_buffer;

    const std::array<float, 24> m_quad_vertices = {
        -1.f,  1.f,  0.f, 1.f,
        -1.f, -1.f,  0.f, 0.f,
         1.f, -1.f,  1.f, 0.f,
        -1.f,  1.f,  0.f, 1.f,
         1.f, -1.f,  1.f, 0.f,
         1.f,  1.f,  1.f, 1.f,
    };
    unsigned int m_quad_VAO;
    unsigned int m_quad_VBO;

    unsigned int m_blurFBO[2], m_blur_buffer[2];

};

#endif
