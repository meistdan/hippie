#version 330 core

// Vertex position.
layout (location = 0) in vec4 posAttrib;

// Input tetxure coordinates.
layout (location = 3) in vec2 texAttrib;

// Output tetxure coordinates.
out vec2 texCoord; 

// A main entry point of the shader program.
void main() {
    gl_Position = posAttrib;
    texCoord = texAttrib;
}