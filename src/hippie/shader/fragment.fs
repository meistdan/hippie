#version 330 core

// Texture coordinates.
in vec2 texCoord;

/// Fragment color.
layout (location = 0) out vec4 FragColor;

// A texture sampler.
uniform sampler2D texSampler;

// A main entry of the shader program.
void main() {
    FragColor = texture2D(texSampler, texCoord);
}
