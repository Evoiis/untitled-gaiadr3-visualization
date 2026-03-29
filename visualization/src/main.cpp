#include <iostream>

#include "node.hpp"
#include "star_data.hpp"

int main() {

    auto shared_stars = std::make_shared<SharedStars>();
    int port = 5656; // Launch Param?

    Node node{shared_stars, port};

    node.request_gaia_data();

    return 0;
}
