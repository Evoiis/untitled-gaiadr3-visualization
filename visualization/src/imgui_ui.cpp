#include "imgui_ui.hpp"

ImguiUI::ImguiUI(){

}

void ImguiUI::render_ui(){

}


bool ImguiUI::calculate_label_position(glm::mat4 mvp, glm::vec3 cam_pos, StarData star, float width, float height, ImVec2 & result){
    if(glm::length2(cam_pos - star.position_xyz) > m_far_label_clip * m_far_label_clip){
        return false;
    }

    // Transform star position to clip space
    auto clip_space_pos = mvp * glm::vec4(star.position_xyz, 1.);
    
    // Check if star is behind the camera
    if(clip_space_pos.w <= 0){
        return false;
    }
    
    // Translate to normalized device coordinates
    glm::vec3 ndc = glm::vec3(clip_space_pos) / clip_space_pos.w;

    // Check if outside of FOV
    if(ndc.x > 1 || ndc.x < -1 || ndc.y > 1 || ndc.y < -1){
        return false;
    }

    
    result.x = (ndc.x + 1) / 2 * width;
    result.y = (1 - ndc.y) / 2 * height;    // Y is flipped in Imgui so (1 - y)

    return true;
}
