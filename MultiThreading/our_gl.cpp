#include <algorithm>
#include <mutex>   
#include <memory>
#include "our_gl.h"

static std::vector<std::unique_ptr<std::mutex>> g_row_mutexes;

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

// 單 triangle、單 thread 執行，但可被多個 thread 同時呼叫
void rasterize_singlethread(const Triangle &clip,
                            const IShader &shader,
                            TGAImage &framebuffer) {
    vec4 ndc[3]    = { clip[0]/clip[0].w, clip[1]/clip[1].w, clip[2]/clip[2].w };
    vec2 screen[3] = { (Viewport*ndc[0]).xy(), (Viewport*ndc[1]).xy(), (Viewport*ndc[2]).xy() };

    mat<3,3> ABC = {{ {screen[0].x, screen[0].y, 1.},
                      {screen[1].x, screen[1].y, 1.},
                      {screen[2].x, screen[2].y, 1.} }};
    if (ABC.det() < 1) return; // backface culling

    auto [bbminx,bbmaxx] = std::minmax({screen[0].x, screen[1].x, screen[2].x});
    auto [bbminy,bbmaxy] = std::minmax({screen[0].y, screen[1].y, screen[2].y});

    const int xmin = std::max<int>(bbminx, 0);
    const int xmax = std::min<int>(bbmaxx, framebuffer.width()  - 1);
    const int ymin = std::max<int>(bbminy, 0);
    const int ymax = std::min<int>(bbmaxy, framebuffer.height() - 1);

    mat<3,3> ABC_invT = ABC.invert_transpose();

    for (int x = xmin; x <= xmax; ++x) {
        for (int y = ymin; y <= ymax; ++y) {
            vec3 bc_screen = ABC_invT * vec3{(double)x, (double)y, 1.};
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;

            vec3 bc_clip = { bc_screen.x/clip[0].w,
                             bc_screen.y/clip[1].w,
                             bc_screen.z/clip[2].w };
            bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);

            double z = bc_screen * vec3{ ndc[0].z, ndc[1].z, ndc[2].z };

            // 先算出 fragment 顏色（不鎖）
            auto [discard, color] = shader.fragment(bc_clip);
            if (discard) continue;

            const int idx = x + y * framebuffer.width();
            
            // b
            // // 只在要更新 zbuffer / framebuffer 時鎖該列
            // {
            //     std::lock_guard<std::mutex> lock(g_zbuffer_mutex);
            //     if (z <= zbuffer[idx]) continue;
            //     zbuffer[idx] = z;
            //     framebuffer.set(x, y, color);
            // }
            
            // a
            // 實作行級別鎖定 (Row-Level Locking)
            {
                // 使用 *g_row_mutexes[y] 取得該行鎖的實體
                std::lock_guard<std::mutex> lock(*g_row_mutexes[y]); 
                
                // Z 測試、Z Buffer 更新和 Framebuffer 寫入仍在鎖的保護下
                if (z <= zbuffer[idx]) continue;
                zbuffer[idx] = z;
                framebuffer.set(x, y, color);
            } // 鎖在此處自動釋放 (RAII)
        }
    }
}

// 原始單執行緒 rasterize（作為 baseline / 參考）
void rasterize(const Triangle &clip, const IShader &shader, TGAImage &framebuffer) {
    vec4 ndc[3]    = { clip[0]/clip[0].w, clip[1]/clip[1].w, clip[2]/clip[2].w };                // normalized device coordinates
    vec2 screen[3] = { (Viewport*ndc[0]).xy(), (Viewport*ndc[1]).xy(), (Viewport*ndc[2]).xy() }; // screen coordinates

    mat<3,3> ABC = {{ {screen[0].x, screen[0].y, 1.},
                      {screen[1].x, screen[1].y, 1.},
                      {screen[2].x, screen[2].y, 1.} }};
    if (ABC.det()<1) return; // backface culling + discarding triangles that cover less than a pixel

    auto [bbminx,bbmaxx] = std::minmax({screen[0].x, screen[1].x, screen[2].x}); // bounding box for the triangle
    auto [bbminy,bbmaxy] = std::minmax({screen[0].y, screen[1].y, screen[2].y}); // defined by its top left and bottom right corners

    mat<3,3> ABC_invT = ABC.invert_transpose();

    for (int x = std::max<int>(bbminx, 0);
             x <= std::min<int>(bbmaxx, framebuffer.width()-1); x++) {         // clip the bounding box by the screen
        for (int y = std::max<int>(bbminy, 0);
                 y <= std::min<int>(bbmaxy, framebuffer.height()-1); y++) {
            vec3 bc_screen = ABC_invT * vec3{static_cast<double>(x),
                                              static_cast<double>(y), 1.}; // barycentric coordinates of {x,y} w.r.t the triangle
            vec3 bc_clip   = { bc_screen.x/clip[0].w,
                               bc_screen.y/clip[1].w,
                               bc_screen.z/clip[2].w };     // check wiki for perspective correction
            bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);
            if (bc_screen.x<0 || bc_screen.y<0 || bc_screen.z<0) continue; // negative barycentric coordinate => the pixel is outside the triangle
            double z = bc_screen * vec3{ ndc[0].z, ndc[1].z, ndc[2].z };   // linear interpolation of the depth
            if (z <= zbuffer[x+y*framebuffer.width()]) continue;   // discard fragments that are too deep w.r.t the z-buffer
            auto [discard, color] = shader.fragment(bc_clip);
            if (discard) continue;                                 // fragment shader can discard current fragment
            zbuffer[x+y*framebuffer.width()] = z;                  // update the z-buffer
            framebuffer.set(x, y, color);                          // update the framebuffer
        }
    }
}
