#include <zmq.hpp>
#include <google/protobuf/message.h>
#include <iostream>
#include <chrono>

#include "star_data.pb.h"
#include "node.hpp"


Node::Node(std::shared_ptr<SharedStars> shared_stars_ptr, int port){
    m_socket = zmq::socket_t(m_ctx, zmq::socket_type::req);
    m_server_address = "tcp://localhost:" + std::to_string(port);

    m_shared_stars_ptr = shared_stars_ptr;
}

int Node::request_gaia_data(){
    try{
        m_socket.connect(m_server_address);
    }catch (const zmq::error_t& e){
        std::cerr << "Connect failed: " << e.what() << std::endl;
        return 1;
    }

    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto now_seconds = now.count() * std::chrono::system_clock::period::num / std::chrono::system_clock::period::den;

    mwm_msgs::DataRequest data_req;
    data_req.set_timestamp(now_seconds);
    data_req.set_node_name("Visualization");
    data_req.set_star_data_type(mwm_msgs::StarDataType::STAR_DATA);

    std::string serialized_data_req;
    if(!data_req.SerializeToString(&serialized_data_req)){
        std::cerr << "Failed to serialize request!" << std::endl;
        return 1;
    }
    

    // Send message
    std::cout << "Sending request..." << std::endl;
    zmq::message_t request(serialized_data_req.data(), serialized_data_req.size());
    m_socket.send(request, zmq::send_flags::none);

    // Receive reply
    zmq::message_t reply;
    try {
        m_socket.recv(reply, zmq::recv_flags::none);
    } catch (const zmq::error_t& e) {
        std::cerr << "recv failed: " << e.what() << std::endl;
        return 1;
    }

    // Parse Stars message
    mwm_msgs::Stars stars;
    if (!stars.ParseFromArray(reply.data(), reply.size())) {
        std::cerr << "Failed to parse stars message!" << std::endl;
        return 1;
    }

    // Process reply
    std::cout << "Received " << stars.stars().size() << " stars at timestamp: "
        << stars.timestamp() << std::endl;
    
    m_shared_stars_ptr->set(reformat_data(stars));
    return 0;
}

StarDataPtr Node::reformat_data(const mwm_msgs::Stars& stars){
    auto new_star_data_ptr = std::make_shared<StarData>();


    for (const auto& [id, proto_star] : stars.stars()) {
        new_star_data_ptr->star_positions.emplace_back(
            proto_star.pos_x(), proto_star.pos_y(), proto_star.pos_z()
        );

        new_star_data_ptr->star_colors_rgb.emplace_back(
            proto_star.color_r(), proto_star.color_g(), proto_star.color_b()
        );

        new_star_data_ptr->star_brightness.emplace_back(proto_star.brightness());
        new_star_data_ptr->star_sizes.emplace_back(proto_star.size());

        if(proto_star.has_name()){
            new_star_data_ptr->star_names.emplace_back(proto_star.name());
        }else{
            new_star_data_ptr->star_names.emplace_back(std::to_string(id));
        }
    }
    return new_star_data_ptr;
}
