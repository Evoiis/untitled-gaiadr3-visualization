// #include <GL/glew.h> 
// #include <GLFW/glfw3.h>
// #include <glm/glm.hpp>
// #include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtc/type_ptr.hpp>
// #include <glm/gtx/norm.hpp>

// #include <iostream>
// #include <chrono>
// #include <thread>
// #include <vector>

// #include "imgui.h"
// #include "imgui_impl_glfw.h"
// #include "imgui_impl_opengl3.h"

// #include "shader.hpp"


// struct Camera{
//     glm::vec3 pos   = glm::vec3(0.0f, 0.0f,  3.0f);
//     glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f); // direction camera is pointing
//     glm::vec3 up    = glm::vec3(0.0f, 1.0f,  0.0f); // which way is up for the camera, can edit to roll the camera
// };

// void framebuffer_size_callback(GLFWwindow* window, int width, int height)
// {
//     glViewport(0, 0, width, height);
//     // pointScale ; // Can scale pointscale if we need here
// }  

// void processInput(GLFWwindow *window)
// {
//     if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
//         glfwSetWindowShouldClose(window, true);
//     }

// }

// void processCameraInput(GLFWwindow *window, Camera& cam, float delta_time){
//     float cameraSpeed = 10.f * delta_time;
//     if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
//         cameraSpeed = 40.f * delta_time;
//     }
//     if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
//         cam.pos += cameraSpeed * cam.front;
//     }
//     if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
//         cam.pos -= cameraSpeed * cam.front;
//     }
//     if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
//         // std::cout << "A" << std::endl;
//         cam.pos -= glm::normalize(glm::cross(cam.front, cam.up)) * cameraSpeed;
//     }
//     if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
//         // std::cout << "D" << std::endl;
//         cam.pos += glm::normalize(glm::cross(cam.front, cam.up)) * cameraSpeed;
//     }
// }

// float lastX = 400, lastY = 300;
// float yaw = -90.0f;
// float pitch = 0.f;
// bool firstMouse = true;
// float pointScale;

// void processMouseInput(GLFWwindow * window, double xpos, double ypos){

//     if(ImGui::GetIO().WantCaptureMouse) return; // let ImGui have mouse input

//     if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS){
//         firstMouse = true;
//         return;
//     }
    
//     // Prevents pop in jerk on first mouse
//     if (firstMouse)
//     {
//         lastX = xpos;
//         lastY = ypos;
//         firstMouse = false;
//     }
//     float xoffset = xpos - lastX;
//     float yoffset = lastY - ypos; // reversed since y-coordinates range from bottom to top
//     lastX = xpos;
//     lastY = ypos;

//     const float sensitivity = 0.1f;
//     xoffset *= sensitivity;
//     yoffset *= sensitivity;

//     yaw   += xoffset;
//     pitch += yoffset;
//     if(pitch > 89.0f){
//         pitch =  89.0f;
//     }
//     if(pitch < -89.0f){
//         pitch = -89.0f;
//     }
// }


// struct StarVertex {
//     glm::vec3 position;
//     float magnitude;
//     float color;
// };

// struct StarMeta {
//     std::string name;
// };

// // Star Label Params
// float far_clip = 100.f;  // tweak if necessary, maybe tweak with star magnitude in the future?
// // float near_clip = 1.f;

// bool calculate_label_position(glm::mat4 mvp, glm::vec3 cam_pos, StarVertex star, float width, float height, ImVec2 & result){
//     if(glm::length2(cam_pos - star.position) > far_clip * far_clip){
//         return false;
//     }

//     // Transform star position to clip space
//     auto clip_space_pos = mvp * glm::vec4(star.position, 1.);
    
//     // Check if star is behind the camera
//     if(clip_space_pos.w <= 0){
//         return false;
//     }
    
//     // Translate to normalized device coordinates
//     glm::vec3 ndc = glm::vec3(clip_space_pos) / clip_space_pos.w;

//     // Check if outside of FOV
//     if(ndc.x > 1 || ndc.x < -1 || ndc.y > 1 || ndc.y < -1){
//         return false;
//     }

    
//     result.x = (ndc.x + 1) / 2 * width;
//     result.y = (1 - ndc.y) / 2 * height;    // Y is flipped in Imgui so (1 - y)

//     return true;
// }

int main(){
//     glfwInit();
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//     glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
//     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//     //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // TODO macos check

//     float width = 1280; // Parameter?
//     float height = 720;

//     GLFWwindow* window = glfwCreateWindow(width, height, "LearnOpenGL", NULL, NULL);
//     if (window == NULL)
//     {
//         std::cout << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return -1;
//     }

//     glfwMakeContextCurrent(window);
//     glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

//     glewExperimental = GL_TRUE;
//     glewInit();
//     glClearColor(0.f, 0.f, 0.f, 1.0f);

//     // FBO Setup ----------------------
//     unsigned int hdrFBO, colorBuffer;
//     glGenFramebuffers(1, &hdrFBO);
//     glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);

//     // Color texture attachment
//     glGenTextures(1, &colorBuffer);
//     glBindTexture(GL_TEXTURE_2D, colorBuffer);
//     // Allocate texture memory
//     // GL_RGBA16F allows values > 1.0 for over-bright stars
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
//     // Use lerp to sample the texture when scaling
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//     // Clamp edges
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//     // Attach the texture to the FBO
//     glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorBuffer, 0);

//     if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
//         std::cout << "FBO not complete" << std::endl;

//     // Unbind hdr
//     glBindFramebuffer(GL_FRAMEBUFFER, 0);

//     // Bright FBO Setup -------------------------------------
//     unsigned int brightFBO, brightBuffer;
//     glGenFramebuffers(1, &brightFBO);
//     glBindFramebuffer(GL_FRAMEBUFFER, brightFBO);

//     glGenTextures(1, &brightBuffer);
//     glBindTexture(GL_TEXTURE_2D, brightBuffer);
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//     // Clamp edges
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//     glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brightBuffer, 0);

//     if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
//         std::cout << "Bright FBO not complete" << std::endl;

//     glBindFramebuffer(GL_FRAMEBUFFER, 0);

//     // Setup quad for texture ----------------------------
//     float quadVerts[] = {
//         // pos         // uv
//         -1.f,  1.f,   0.f, 1.f,
//         -1.f, -1.f,   0.f, 0.f,
//         1.f, -1.f,   1.f, 0.f,

//         -1.f,  1.f,   0.f, 1.f,
//         1.f, -1.f,   1.f, 0.f,
//         1.f,  1.f,   1.f, 1.f,
//     };

//     unsigned int quadVAO, quadVBO;
//     glGenVertexArrays(1, &quadVAO);
//     glGenBuffers(1, &quadVBO);
//     glBindVertexArray(quadVAO);
//     glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
//     glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
//     glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);
//     glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
//     glEnableVertexAttribArray(1);

//     // Ping Pong FBO Setup --------------

//     unsigned int blurFBO[2], blurBuffer[2];
//     glGenFramebuffers(2, blurFBO);
//     glGenTextures(2, blurBuffer);
//     for(int i = 0; i < 2; i++){
//         glBindFramebuffer(GL_FRAMEBUFFER, blurFBO[i]);
//         glBindTexture(GL_TEXTURE_2D, blurBuffer[i]);
//         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
//         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//         // Clamp edges
//         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//         glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blurBuffer[i], 0);
//     }
//     glBindFramebuffer(GL_FRAMEBUFFER, 0);

//     // IMGUI Init
//     IMGUI_CHECKVERSION();
//     ImGui::CreateContext();
//     ImGui::StyleColorsDark();
//     ImGui_ImplGlfw_InitForOpenGL(window, true);
//     ImGui_ImplOpenGL3_Init("#version 330");

    // Init Shader Program
    // Shader shader_program(SHADER_DIR "point_sprites.vs", SHADER_DIR "point_sprites.fs");
    // shader_program.use();
    // Init Point Scale - controls star size
    // pointScale = 2.0f;  // Parameter
    // glUniform1f(glGetUniformLocation(shader_program.m_ID, "uPointScale"), height * pointScale);

    // Init Bright Shader
    // Shader brightness_shader(SHADER_DIR "screen.vs", SHADER_DIR "bright.fs");
    // brightness_shader.use();

    // float threshold = 0.3f; // Parameter or controlled by star data
    // glUniform1f(glGetUniformLocation(brightness_shader.m_ID, "threshold"), threshold);

    // // Init Blur Shader
    // Shader blur_shader(SHADER_DIR "screen.vs", SHADER_DIR "blur.fs");
    // float blurAmount = 10;  // Parameter

    // // Init Combine Shader
    // Shader combine_shader(SHADER_DIR "screen.vs", SHADER_DIR "combine.fs");
    // combine_shader.use();
    // // Which texture unit to sample from
    // glUniform1i(glGetUniformLocation(combine_shader.m_ID, "scene"), 0);
    // glUniform1i(glGetUniformLocation(combine_shader.m_ID, "bloomBlur"), 1);

    // float bloomStrength = 1.0f; // Parameter or controlled by star data
    // glUniform1f(glGetUniformLocation(combine_shader.m_ID, "bloomStrength"), bloomStrength);

    // Must match stars in stars data below
    std::vector<StarMeta> star_meta_data = {
        {"Sirius"},
        {"Vega"},
        {"Sol"},
        {"Ran"},
        {"Proxima Centauri"},
        {"Altair"},
        {"Fomalhaut"},
        {"Pollux"},
        {"Arcturus"},
        {"Betelgeuse"},
        {"Rigel"},
        {"Deneb"},
        {"Spica"},
        {"Antares"},
        {"Aldebaran"},
    };

    // Init Stars
    std::vector<StarVertex> stars = {
        // position, magnitude, color
        {{ 0.0f,  1.0f,  -5.0f}, -1.46f, 0.5f}, // Sirius-like,  blue-white
        {{ 1.5f,  0.5f,  -8.0f},  0.45f, 0.71f}, // Vega-like,    white
        {{ 0.5f,  0.5f,  -6.f},   2.5f, 3.5f}, // sun-like,     yellow-white
        {{ 3.0f, -0.5f, -20.0f},  2.80f, 3.50f}, // K-type,       orange
        {{-2.5f,  1.5f, -30.0f},  4.10f, 4.20f}, // M-type,       red
        // mid-range fill
        {{ 2.0f,  3.0f, -15.0f},  2.10f, 0.50f},
        {{-3.0f, -1.0f, -18.0f},  3.20f, 2.10f},
        {{ 0.5f, -2.0f, -25.0f},  3.80f, 1.40f},
        {{-1.5f,  0.5f, -10.0f},  1.90f, 0.90f},
        {{ 4.0f,  1.0f, -35.0f},  4.50f, 3.80f},
        // dim background
        {{ 1.0f,  4.0f, -40.0f},  5.50f, 1.20f},
        {{-4.0f,  2.0f, -50.0f},  -2.f, 4.0f},
        {{ 2.5f, -3.0f, -45.0f},  5.80f, 2.90f},
        {{-0.5f, -4.0f, -60.0f},  6.20f, 1.80f},
        {{ 5.0f,  0.5f, -55.0f},  5.90f, 3.20f},
    };


    // Init Vertex Array Object
    // unsigned int VAO;
    // glGenVertexArrays(1, &VAO);    
    // glBindVertexArray(VAO);

    // // Generate Vertex Buffer Object
    // GLuint VBO;
    // glGenBuffers(1, &VBO);
    // glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // // Copy vertices data into buffer's memory
    // glBufferData(GL_ARRAY_BUFFER, stars.size() * sizeof(StarVertex), stars.data(), GL_STATIC_DRAW);

    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)offsetof(StarVertex, position));
    // glEnableVertexAttribArray(0);

    // glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)offsetof(StarVertex, magnitude));
    // glEnableVertexAttribArray(1);

    // glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)offsetof(StarVertex, color));
    // glEnableVertexAttribArray(2);



    // Camera (view matrix)
    // glm::mat4 view;
    // Camera cam;
    // glm::vec3 direction;
    
    // direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    // direction.y = sin(glm::radians(pitch));
    // direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    // cam.front = glm::normalize(direction);
    
    // view = glm::lookAt(cam.pos, cam.pos + cam.front, cam.up);


    // MVP Composite

    // don't need this because my star positions are already in world space
    // local -> world space
    // glm::mat4 model = glm::mat4(1.0f);
    // model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    // don't need model, stars already in world space
    
    // view space -> clip space
    // glm::mat4 projection;
    // glm::perspective Parameters: FOV, Aspect Ratio (w/h), near plane, far plane
    // Parameter: near and far plane values
    // projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.05f, 500.0f);

    // glm::mat4 mvp_composite = projection * view;
    // int compositeLoc = glGetUniformLocation(shader_program.m_ID, "mvp_composite");
    // glUniformMatrix4fv(compositeLoc, 1, GL_FALSE, glm::value_ptr(mvp_composite));

    // Depth On
    // glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // additive — stars glow through each other
    glDepthMask(GL_FALSE); // Prevent z-fighting at the far end of the scene

    // Controls
    bool pause = false;
    bool space_was_pressed = false;

    
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Hide mouse cursor
    // glfwSetCursorPosCallback(window, processMouseInput);

    // Frame time
    float delta_time = 0.0f;
    float last_frame_time = 0.0f;

    // Mouse positions
    double mouse_xpos, mouse_ypos;

    while (!glfwWindowShouldClose(window))
    {
        float now = (float)glfwGetTime();
        delta_time = now - last_frame_time;
        last_frame_time = now;

        // Controls
        bool space_pressed = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        if(space_pressed && !space_was_pressed){
            pause = !pause;
        }
        space_was_pressed = space_pressed;

        if(pause){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            glfwPollEvents();
            processInput(window);
            continue;
        }

        glfwGetCursorPos(window, &mouse_xpos, &mouse_ypos);
        processMouseInput(window, mouse_xpos, mouse_ypos);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();        
        
        processInput(window);
        
        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cam.front = glm::normalize(direction);
        
        processCameraInput(window, cam, delta_time);
        view = glm::lookAt(cam.pos, cam.pos + cam.front, cam.up);
        
        shader_program.use();

        // Bind FBO and clear buffers
        glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Update MVP Composite
        glm::mat4 mvp_composite = projection * view;
        glUniformMatrix4fv(compositeLoc, 1, GL_FALSE, glm::value_ptr(mvp_composite));

        // Draw Stars
        glEnable(GL_BLEND);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, stars.size());
        glDisable(GL_BLEND);

        // Brightness Shader Pass
        glBindFramebuffer(GL_FRAMEBUFFER, brightFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        brightness_shader.use();
        glBindVertexArray(quadVAO);
        glUniform1i(glGetUniformLocation(brightness_shader.m_ID, "screenTexture"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorBuffer);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Blur Pass
        bool horizontal = true;
        blur_shader.use();

        glUniform1i(glGetUniformLocation(blur_shader.m_ID, "image"), 0);
        glBindFramebuffer(GL_FRAMEBUFFER, blurFBO[0]);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, blurFBO[1]);
        glClear(GL_COLOR_BUFFER_BIT);
        
        // First iteration reads from brightBuffer, every subsequent iteration ping-pongs between the two blur FBOs.
        // After 10 passes the result is in blurBuffer[!horizontal].
        for(int i = 0; i < (int)blurAmount; i++){
            glBindFramebuffer(GL_FRAMEBUFFER, blurFBO[horizontal]);
            glUniform1i(glGetUniformLocation(blur_shader.m_ID, "horizontal"), horizontal);
            glBindTexture(GL_TEXTURE_2D, i == 0 ? brightBuffer : blurBuffer[!horizontal]);
            glBindVertexArray(quadVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            horizontal = !horizontal;
        }

        // Back to FBO 0 (Screen) to draw the result with combine pass
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Combine Pass
        combine_shader.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorBuffer);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, blurBuffer[!horizontal]);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Imgui Draw Gui
        ImGui::SetNextWindowSize(ImVec2(300, 200));
        ImGui::Begin("Debug");
        ImGui::Text("Cam: %.2f %.2f %.2f", cam.pos.x, cam.pos.y, cam.pos.z);
        ImGui::Text("Yaw: %.1f  Pitch: %.1f", yaw, pitch);
        ImGui::Text("Stars: %zu", stars.size());
        ImGui::Text("WantMouse: %d", ImGui::GetIO().WantCaptureMouse);

        // Update Blur Parameters
        ImGui::SliderFloat("Blur Amount", &blurAmount, 1.0f, 100.0f);
        ImGui::Text("BlurAmount: %d", (int)blurAmount);

        ImGui::SliderFloat("Threshold", &threshold, 0.0f, 1.0f);
        brightness_shader.use();
        glUniform1f(glGetUniformLocation(brightness_shader.m_ID, "threshold"), threshold);

        ImGui::SliderFloat("Bloom Strength", &bloomStrength, 0.0f, 3.0f);
        combine_shader.use();
        glUniform1f(glGetUniformLocation(combine_shader.m_ID, "bloomStrength"), bloomStrength);

        // Draw Star Labels
        // GetBackgroundDrawLists, Draws behind GUI, but in front of scene        
        ImVec2 label_position;
        bool should_draw;
        for(int i = 0; i < stars.size(); i++){
            should_draw = calculate_label_position(mvp_composite, cam.pos, stars[i], width, height, label_position);
            if(should_draw){
                ImGui::GetBackgroundDrawList()->AddText(
                    label_position,
                    IM_COL32(255, 255, 255, 255),
                    star_meta_data[i].name.c_str()
                );
            }
        }

        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }  

    std::cout << "Terminating Visualization" << std::endl;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}


