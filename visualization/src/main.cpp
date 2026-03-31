#include <iostream>

#include "node.hpp"
#include "star_data.hpp"

#include "vis.hpp"

int main() {

    auto shared_stars = std::make_shared<SharedStars>();
    int port = 5656; // Launch Param?
    int width = 1280;
    int height = 720;

    float camera_label_close_clip = 100.f;
    float camera_label_far_clip = 1.f;

    Node node{shared_stars, port};
    // node.request_gaia_data();

    Camera cam;
    BloomPipeline bp;
    ImguiUI ui(camera_label_close_clip, camera_label_far_clip);

    Visualization vis(shared_stars, cam, bp, ui, 2.0f, 2,2);

    return 0;
}
