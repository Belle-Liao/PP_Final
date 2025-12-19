#include <algorithm>
#include "our_gl.h"

mat<4,4> ModelView, Viewport, Perspective; // "OpenGL" state matrices
std::vector<double> zbuffer;               // depth buffer

void lookat(const vec3 eye, const vec3 center, const vec3 up) {
    vec3 n = normalized(eye-center);
    vec3 l = normalized(cross(up,n));
    vec3 m = normalized(cross(n, l));
    ModelView = mat<4,4>{{{l.x,l.y,l.z,0}, {m.x,m.y,m.z,0}, {n.x,n.y,n.z,0}, {0,0,0,1}}} *
                mat<4,4>{{{1,0,0,-center.x}, {0,1,0,-center.y}, {0,0,1,-center.z}, {0,0,0,1}}};
}

void init_perspective(const double f) {
    Perspective = {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0, -1/f,1}}};
}

void init_viewport(const int x, const int y, const int w, const int h) {
    Viewport = {{{w/2., 0, 0, x+w/2.}, {0, h/2., 0, y+h/2.}, {0,0,1,0}, {0,0,0,1}}};
}

void init_zbuffer(const int width, const int height) {
    zbuffer = std::vector(width*height, -1000.);
}

// for tile-based
void rasterize_tile(const Triangle &clip, const IShader &shader, TGAImage &framebuffer) {
    int width  = framebuffer.width();
    int height = framebuffer.height();

    // NDC coordinates
    vec4 ndc[3]    = { clip[0]/clip[0].w, clip[1]/clip[1].w, clip[2]/clip[2].w };
    vec2 screen[3] = { (Viewport*ndc[0]).xy(), (Viewport*ndc[1]).xy(), (Viewport*ndc[2]).xy() };

    mat<3,3> ABC = {{ {screen[0].x, screen[0].y, 1.},
                      {screen[1].x, screen[1].y, 1.},
                      {screen[2].x, screen[2].y, 1.} }};
    if (ABC.det()<1) return;

    // Bounding box
    int bbminx = std::max(0, int(std::min({screen[0].x, screen[1].x, screen[2].x})));
    int bbmaxx = std::min(width-1, int(std::max({screen[0].x, screen[1].x, screen[2].x})));
    int bbminy = std::max(0, int(std::min({screen[0].y, screen[1].y, screen[2].y})));
    int bbmaxy = std::min(height-1, int(std::max({screen[0].y, screen[1].y, screen[2].y})));

    // Tile size
    int tile_size = 32;

    // Create thread-local framebuffer and zbuffer
    int num_tiles_x = (bbmaxx - bbminx)/tile_size + 1;
    int num_tiles_y = (bbmaxy - bbminy)/tile_size + 1;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ty = 0; ty < num_tiles_y; ty++) {
        for (int tx = 0; tx < num_tiles_x; tx++) {
            int x_start = bbminx + tx*tile_size;
            int y_start = bbminy + ty*tile_size;
            int x_end   = std::min(x_start + tile_size - 1, bbmaxx);
            int y_end   = std::min(y_start + tile_size - 1, bbmaxy);

            // Local zbuffer + framebuffer for this thread
            std::vector<double> local_z((x_end-x_start+1)*(y_end-y_start+1), -1e10);
            TGAImage local_fb(x_end-x_start+1, y_end-y_start+1, TGAImage::RGB);

            for (int y = y_start; y <= y_end; y++) {
                for (int x = x_start; x <= x_end; x++) {
                    vec3 bc_screen = ABC.invert_transpose() * vec3{double(x), double(y), 1.};
                    if (bc_screen.x<0 || bc_screen.y<0 || bc_screen.z<0) continue;

                    vec3 bc_clip = { bc_screen.x/clip[0].w, bc_screen.y/clip[1].w, bc_screen.z/clip[2].w };
                    bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);

                    double z = bc_screen * vec3{ ndc[0].z, ndc[1].z, ndc[2].z };
                    int lx = x - x_start;
                    int ly = y - y_start;
                    int lidx = lx + ly*(x_end-x_start+1);

                    auto [discard, color] = shader.fragment(bc_clip);
                    if (!discard) {
                        if (z > local_z[lidx]) {
                            local_z[lidx] = z;
                            local_fb.set(lx, ly, color);
                        }
                    }
                }
            }

            // Merge local tile to global framebuffer & zbuffer
            for (int y = y_start; y <= y_end; y++) {
                for (int x = x_start; x <= x_end; x++) {
                    int lx = x - x_start;
                    int ly = y - y_start;
                    int lidx = lx + ly*(x_end-x_start+1);
                    if (local_z[lidx] > zbuffer[x + y*width]) {
                        zbuffer[x + y*width] = local_z[lidx];
                        framebuffer.set(x, y, local_fb.get(lx, ly));
                    }
                }
            }
        }
    }
}

// for face-level
void rasterize_face(const Triangle &clip, const IShader &shader, TGAImage &framebuffer) {
    // Homogeneous division: Clip space -> NDC
    vec4 ndc[3]    = { clip[0]/clip[0].w, clip[1]/clip[1].w, clip[2]/clip[2].w };
    // Viewport transform: NDC -> Screen Coordinates
    vec2 screen[3] = { (Viewport*ndc[0]).xy(), (Viewport*ndc[1]).xy(), (Viewport*ndc[2]).xy() };

    // Setup matrix for barycentric coordinate calculation
    mat<3,3> ABC = {{ {screen[0].x, screen[0].y, 1.}, {screen[1].x, screen[1].y, 1.}, {screen[2].x, screen[2].y, 1.} }};
    // Backface culling + discarding tiny triangles
    if (ABC.det()<1) return; 

    // Calculate Bounding Box
    auto [bbminx,bbmaxx] = std::minmax({screen[0].x, screen[1].x, screen[2].x});
    auto [bbminy,bbmaxy] = std::minmax({screen[0].y, screen[1].y, screen[2].y});
    
    // Iterate over pixels in the bounding box (Parallel loop)
#pragma omp parallel for
    for (int x=std::max<int>(bbminx, 0); x<=std::min<int>(bbmaxx, framebuffer.width()-1); x++) { 
        for (int y=std::max<int>(bbminy, 0); y<=std::min<int>(bbmaxy, framebuffer.height()-1); y++) {
            
            // Calculate screen-space barycentric coordinates (bc_screen)
            vec3 bc_screen = ABC.invert_transpose() * vec3{static_cast<double>(x), static_cast<double>(y), 1.};
            
            // Check if pixel is inside the triangle
            if (bc_screen.x<0 || bc_screen.y<0 || bc_screen.z<0) continue; 

            // Perspective Correction: calculate clip-space barycentric coordinates (bc_clip) 
            // for correct interpolation of attributes (like UVs and Normals)
            vec3 bc_clip    = { bc_screen.x/clip[0].w, bc_screen.y/clip[1].w, bc_screen.z/clip[2].w };
            bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);

            // Interpolate depth Z (NDC Z value)
            double z = bc_screen * vec3{ ndc[0].z, ndc[1].z, ndc[2].z }; 
            
            int idx = x + y*framebuffer.width(); // Shared buffer index

            // Critical Section: Protect shared resource R/W
            // This ensures only one thread performs the Z-Test and writes to the pixel at a time,
            // preventing data races on zbuffer and framebuffer.
            #pragma omp critical (zbuffer_write)
            {
                // Z-Test: If current fragment is closer (Z value is larger), proceed
                if (z > zbuffer[idx]) { 
                    
                    // Call Fragment Shader with perspective-corrected bc_clip
                    auto [discard, color] = shader.fragment(bc_clip);

                    // If not discarded
                    if (!discard) {
                        zbuffer[idx] = z;                      // Update Z-buffer (Write)
                        framebuffer.set(x, y, color);          // Update Framebuffer (Write)
                    }
                }
            } // End of critical section
        }
    }
}
