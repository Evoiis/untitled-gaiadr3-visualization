#ifndef NODE_HPP
#define NODE_HPP

#include <iostream>
#include <memory>

#include <zmq.hpp>
#include <google/protobuf/message.h>

#include "star_data.pb.h"
#include "star_data.hpp"

class Node{
public:

    Node(std::shared_ptr<SharedStars> shared_stars_ptr, int port);

    int request_gaia_data();

private:
    int m_port;
    zmq::context_t m_ctx;
    zmq::socket_t m_socket;
    std::string m_server_address;
    std::shared_ptr<SharedStars> m_shared_stars_ptr;

    StarMapPtr reformat_data(const mwm_msgs::Stars& stars);

};

#endif
