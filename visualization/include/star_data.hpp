#ifndef STARDATA_H
#define STARDATA_H

#include <string>
#include <map>
#include <mutex>
#include <memory>

#include <glm/glm.hpp>


struct StarData {
    glm::vec3 position_xyz; // Parsecs    
    glm::vec3 color_rgb; // 0-255
    
    float brightness;
    float size;
    
    std::string name;
};

using StarMap = std::map<int64_t, StarData>;
using StarMapPtr = std::shared_ptr<StarMap>;
class SharedStars{
public:
    StarMapPtr m_stars;
    std::mutex m_data_lock;

    StarMapPtr get(){
        std::lock_guard<std::mutex> lock(m_data_lock);
        return m_stars;
    }

    void set(StarMapPtr new_ptr){
        std::lock_guard<std::mutex> lock(m_data_lock);
        m_stars = new_ptr;
    }

};


#endif
