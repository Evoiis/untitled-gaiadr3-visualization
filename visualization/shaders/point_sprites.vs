#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aMagnitude;
layout(location = 2) in float aColorIndex; // Gaia BP-RP, roughly 0.0 (blue) to 4.0 (red)

uniform mat4 mvp_composite;
uniform float uPointScale;

out float vColorIndex;
out float vBrightness;

void main() {
    vec4 pos = mvp_composite * vec4(aPos, 1.0);
    gl_Position = pos;

    vColorIndex = aColorIndex;

    // lower magnitude = brighter star
    vBrightness = clamp(1.0 - (aMagnitude / 6.5), 0.0, 1.0);
    gl_PointSize = clamp((uPointScale / pos.w) * vBrightness * vBrightness * 4, 1.0, 512.0);
}
