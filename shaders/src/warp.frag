#version 450
// Inverse-radial warp (pincushion when k > 0) for the xrizer experiment.
// Reuses overlay.vert's fullscreen quad + texBounds-mapped texCoord varying;
// this fragment stage applies r' = r*(1 + k*r^2) around the FOV-derived
// center and samples outward (k > 0 magnifies edges).
layout(set = 0, binding = 0) uniform sampler2D src;
layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 color;
// NOTE: first 16 bytes (texBounds) are consumed by the vertex stage; the
// fragment stage reads the full 32-byte push range. Offsets must match the
// pipeline layout in vulkan.rs (8 floats: uMin,uMax,vMin,vMax,cx,cy,k,aspect).
layout(push_constant, std430) uniform pc {
	vec4 texBounds; // uMin, uMax, vMin, vMax (vertex use; padding here)
	vec2 center;    // optical center in texCoord 0-1 space
	float k;        // strength: >0 pincushion, <0 barrel, 0 off
	float aspect;   // extent.width / extent.height
};

void main() {
	vec2 p = vec2((texCoord.x - center.x) * aspect, texCoord.y - center.y);
	float r2 = dot(p, p);
	vec2 pw = p * (1.0 + k * r2);
	vec2 uv = vec2(pw.x / aspect + center.x, pw.y + center.y);
	color = texture(src, uv);
}
