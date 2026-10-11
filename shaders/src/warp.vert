#version 450
// Fullscreen quad for the xrizer warp experiment. Identical mapping to
// overlay.vert, but declares the full 9-float push-constant block so both
// stages agree on layout (see warp.frag).
layout(location = 0) out vec2 outTexCoord;
layout(push_constant, std430) uniform pc {
	vec4 texBounds; // uMin, uMax, vMin, vMax
	vec2 center;    // optical center (fragment use)
	float k;        // strength (fragment use)
	float aspect;   // width/height (fragment use)
	float zoom;     // uniform scale (fragment use)
};

vec2 positions[4] = vec2[](
	vec2(-1.0, -1.0),
	vec2(-1.0, 1.0),
	vec2(1.0, -1.0),
	vec2(1.0, 1.0)
);

vec2 texCoords[4] = vec2[](
	texBounds.xz,
	texBounds.xw,
	texBounds.yz,
	texBounds.yw
);

void main() {
	gl_Position = vec4(positions[gl_VertexIndex], 0.0f, 1.0f);
	outTexCoord = texCoords[gl_VertexIndex];
}
