#version 330 core
in float vColorIndex;
in float vBrightness;

out vec4 fragColor;

vec3 starColor(float bprp) {
    float t = clamp(bprp / 4.0, 0.0, 1.0);
    vec3 hot  = vec3(0.7, 0.85, 1.0);   // blue-white
    vec3 mid  = vec3(1.0, 1.0,  0.95);  // white
    vec3 warm = vec3(1.0, 0.85, 0.5);   // yellow
    vec3 cool = vec3(1.0, 0.4,  0.15);  // red-orange

    if (t < 0.33)
        return mix(hot, mid, t * 3.0);
    else if (t < 0.66)
        return mix(mid, warm, (t - 0.33) * 3.0);
    else
        return mix(warm, cool, (t - 0.66) * 3.0);
}

void main() {
    vec2 uv = gl_PointCoord - vec2(0.5);
    float d = length(uv);
    if (d > 0.5) discard;

    vec3 color = starColor(vColorIndex);

    // tight bright core
    float core  = exp(-d * 20.0) * vBrightness;

    // medium halo
    float halo  = exp(-d * 6.0) * vBrightness * .4;

    // wide soft glow — only on bright stars
    float glow = exp(-d * 2.) * vBrightness * vBrightness * 0.1;

    float totalBrightness = core + halo + glow;
    // float totalBrightness = core * 2.; // Can grow size & brightness with multiplication

    float alpha = clamp(totalBrightness, 0.0, 1.0);

    fragColor = vec4(color * totalBrightness, alpha);
}

    // float totalBrightness = core * 20.;