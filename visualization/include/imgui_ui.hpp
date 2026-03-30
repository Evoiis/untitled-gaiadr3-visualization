#ifndef IMGUI_UI_H
#define IMGUI_UI_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "star_data.hpp"

class ImguiUI{
public:

    ImguiUI(float far_label_clip);

    void render_ui();

private:
    float m_far_label_clip;

    bool ImguiUI::calculate_label_position(glm::mat4 mvp, glm::vec3 cam_pos, StarData star, float width, float height, ImVec2 & result);

};

#endif
