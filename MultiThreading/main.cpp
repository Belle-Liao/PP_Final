#include "our_gl.h"
#include "model.h"
#include "cycle_timer.h"
#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

extern mat<4,4> ModelView, Perspective; // "OpenGL" state matrices
extern mat<4,4> Viewport;
extern std::vector<double> zbuffer;     // the depth buffer

unsigned int GLOBAL_THREAD_COUNT = 20;

struct PhongShader : IShader {
    const Model &model;
    vec4 l;              // light direction in eye coordinates
    vec2  varying_uv[3]; // triangle uv coordinates, written by the vertex shader, read by the fragment shader
    vec4 varying_nrm[3]; // normal per vertex to be interpolated by the fragment shader
    vec4 tri[3];         // triangle in view coordinates

    PhongShader(const vec3 light, const Model &m) : model(m) {
        l = normalized((ModelView*vec4{light.x, light.y, light.z, 0.})); // transform the light vector to view coordinates
    }

    virtual vec4 vertex(const int face, const int vert) {
        varying_uv[vert]  = model.uv(face, vert);
        varying_nrm[vert] = ModelView.invert_transpose() * model.normal(face, vert);
        vec4 gl_Position = ModelView * model.vert(face, vert);
        tri[vert] = gl_Position;
        return Perspective * gl_Position;                         // in clip coordinates
    }

    virtual std::pair<bool,TGAColor> fragment(const vec3 bar) const {
        mat<2,4> E = { tri[1]-tri[0], tri[2]-tri[0] };
        mat<2,2> U = { varying_uv[1]-varying_uv[0], varying_uv[2]-varying_uv[0] };
        mat<2,4> T = U.invert() * E;
        mat<4,4> D = {normalized(T[0]),  // tangent vector
                      normalized(T[1]),  // bitangent vector
                      normalized(varying_nrm[0]*bar[0] + varying_nrm[1]*bar[1] + varying_nrm[2]*bar[2]), // interpolated normal
                      {0,0,0,1}}; // Darboux frame
        vec2 uv = varying_uv[0] * bar[0] + varying_uv[1] * bar[1] + varying_uv[2] * bar[2];
        vec4 n = normalized(D.transpose() * model.normal(uv));
        vec4 r = normalized(n * (n * l)*2 - l);                   // reflected light direction
        double ambient  = .4;                                     // ambient light intensity
        double diffuse  = 1.*std::max(0., n * l);                 // diffuse light intensity
        double specular = (.5+2.*sample2D(model.specular(), uv)[0]/255.) * std::pow(std::max(r.z, 0.), 35);  // specular intensity, note that the camera lies on the z-axis (in eye coordinates), therefore simple r.z, since (0,0,1)*(r.x, r.y, r.z) = r.z
        TGAColor gl_FragColor = sample2D(model.diffuse(), uv);
        for (int channel : {0,1,2})
            gl_FragColor[channel] = std::min<int>(255, gl_FragColor[channel]*(ambient + diffuse + specular));
        return {false, gl_FragColor};                             // do not discard the pixel
    }
};

void render_model_face_parallel(const Model &model,
                                const vec3  &light,
                                TGAImage    &framebuffer,
                                int          num_threads) {
    const int nfaces = model.nfaces();
    if (nfaces == 0 || num_threads <= 1) {
        // fallback：直接單執行緒
        PhongShader shader(light, model);
        for (int f = 0; f < nfaces; ++f) {
            Triangle clip = { shader.vertex(f,0),
                              shader.vertex(f,1),
                              shader.vertex(f,2) };
            rasterize_singlethread(clip, shader, framebuffer);
        }
        return;
    }

    const int faces_per_thread = (nfaces + num_threads - 1) / num_threads;
    std::vector<std::thread> workers;

    for (int tid = 0; tid < num_threads; ++tid) {
        int start = tid * faces_per_thread;
        int end   = std::min(start + faces_per_thread, nfaces);
        if (start >= end) break;

        workers.emplace_back([start, end, &model, &light, &framebuffer]() {
            PhongShader shader(light, model); // thread-local shader
            for (int f = start; f < end; ++f) {
                Triangle clip = { shader.vertex(f,0),
                                  shader.vertex(f,1),
                                  shader.vertex(f,2) };
                rasterize_singlethread(clip, shader, framebuffer);
            }
        });
    }

    for (auto &t : workers) t.join();
}

// 每個 face 的預先計算資料（vertex + screen-space 幾何）
struct FacePrecomp {
    Triangle clip;    // clip-space coordinates (Perspective * ModelView * vert)
    vec4     ndc[3];  // normalized device coordinates = clip / w
    vec2     screen[3]; // screen-space (Viewport * ndc).xy()
    mat<3,3> ABC;       // for barycentric
    mat<3,3> ABC_invT;  // inverse-transpose of ABC
    int xmin, xmax;
    int ymin, ymax;
    bool valid;
};

// tile-based 多執行緒渲染一個 model
void render_model_tiled(const Model &model,
                        const vec3  &light,
                        TGAImage    &framebuffer,
                        int          tile_size,
                        int          num_threads) {
    const int width  = framebuffer.width();
    const int height = framebuffer.height();
    const int nfaces = model.nfaces();

    if (nfaces == 0) return;

    // --- Step 1: per-face shader 與幾何前處理（單執行緒） ---
    std::vector<PhongShader> shaders;
    shaders.reserve(nfaces);

    std::vector<FacePrecomp> faces(nfaces);

    for (int f = 0; f < nfaces; ++f) {
        shaders.emplace_back(light, model);
        PhongShader &shader = shaders.back();

        // 呼叫一次 vertex，取得 clip-space 三頂點
        Triangle clip = {
            shader.vertex(f, 0),
            shader.vertex(f, 1),
            shader.vertex(f, 2)
        };

        // NOTE: Triangle 在 tinyrenderer 裡實際上是 C-style array，
        // 所以不能直接 faces[f].clip = clip; 要逐元素拷貝：
        for (int i = 0; i < 3; ++i) {
            faces[f].clip[i] = clip[i];
        }

        // NDC 與 screen-space
        for (int i = 0; i < 3; ++i) {
            faces[f].ndc[i]    = faces[f].clip[i] / faces[f].clip[i].w;
            faces[f].screen[i] = (Viewport * faces[f].ndc[i]).xy();
        }

        // Barycentric matrix
        faces[f].ABC = {{
            {faces[f].screen[0].x, faces[f].screen[0].y, 1.},
            {faces[f].screen[1].x, faces[f].screen[1].y, 1.},
            {faces[f].screen[2].x, faces[f].screen[2].y, 1.}
        }};

        double det = faces[f].ABC.det();
        if (det < 1.0) {
            faces[f].valid = false;
            continue; // backface 或太小的三角形
        }
        faces[f].valid   = true;
        faces[f].ABC_invT = faces[f].ABC.invert_transpose();

        // screen-space bounding box（再 clip 到畫面範圍）
        auto [bbminx, bbmaxx] = std::minmax({faces[f].screen[0].x,
                                             faces[f].screen[1].x,
                                             faces[f].screen[2].x});
        auto [bbminy, bbmaxy] = std::minmax({faces[f].screen[0].y,
                                             faces[f].screen[1].y,
                                             faces[f].screen[2].y});

        int xmin = std::max<int>(bbminx, 0);
        int xmax = std::min<int>(bbmaxx, width  - 1);
        int ymin = std::max<int>(bbminy, 0);
        int ymax = std::min<int>(bbmaxy, height - 1);

        if (xmin > xmax || ymin > ymax) {
            faces[f].valid = false;
            continue;
        }

        faces[f].xmin = xmin;
        faces[f].xmax = xmax;
        faces[f].ymin = ymin;
        faces[f].ymax = ymax;
    }

    // --- Step 2: 建立 tile 網格與 tile -> faces 對應 ---
    const int tiles_x = (width  + tile_size - 1) / tile_size;
    const int tiles_y = (height + tile_size - 1) / tile_size;
    const int total_tiles = tiles_x * tiles_y;

    std::vector<std::vector<int>> tile_faces(total_tiles);

    for (int f = 0; f < nfaces; ++f) {
        if (!faces[f].valid) continue;

        const FacePrecomp &face = faces[f];

        int tx_min = face.xmin / tile_size;
        int tx_max = face.xmax / tile_size;
        int ty_min = face.ymin / tile_size;
        int ty_max = face.ymax / tile_size;

        tx_min = std::max(0, std::min(tx_min, tiles_x - 1));
        tx_max = std::max(0, std::min(tx_max, tiles_x - 1));
        ty_min = std::max(0, std::min(ty_min, tiles_y - 1));
        ty_max = std::max(0, std::min(ty_max, tiles_y - 1));

        for (int ty = ty_min; ty <= ty_max; ++ty) {
            for (int tx = tx_min; tx <= tx_max; ++tx) {
                int tile_id = ty * tiles_x + tx;
                tile_faces[tile_id].push_back(f);
            }
        }
    }

    // --- Step 3: 多執行緒處理 tiles ---
    if (total_tiles == 0) return;

    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 4;
    }
    num_threads = std::min(num_threads, total_tiles);

    const int tiles_per_thread = (total_tiles + num_threads - 1) / num_threads;
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int tid = 0; tid < num_threads; ++tid) {
        int start_tile = tid * tiles_per_thread;
        int end_tile   = std::min(start_tile + tiles_per_thread, total_tiles);

        if (start_tile >= end_tile) break;

        workers.emplace_back([&, start_tile, end_tile, width, height, tiles_x, tiles_y, tile_size]() {
            for (int tile_id = start_tile; tile_id < end_tile; ++tile_id) {
                const auto &face_indices = tile_faces[tile_id];
                if (face_indices.empty()) continue;

                int tx = tile_id % tiles_x;
                int ty = tile_id / tiles_x;

                int tile_xmin = tx * tile_size;
                int tile_xmax = std::min(tile_xmin + tile_size - 1, width  - 1);
                int tile_ymin = ty * tile_size;
                int tile_ymax = std::min(tile_ymin + tile_size - 1, height - 1);

                for (int f_idx : face_indices) {
                    const FacePrecomp &face = faces[f_idx];
                    const PhongShader &shader = shaders[f_idx];

                    // 這個面在此 tile 的交集區域
                    int xmin = std::max(face.xmin, tile_xmin);
                    int xmax = std::min(face.xmax, tile_xmax);
                    int ymin = std::max(face.ymin, tile_ymin);
                    int ymax = std::min(face.ymax, tile_ymax);

                    if (xmin > xmax || ymin > ymax) continue;

                    const mat<3,3> &ABC_invT = face.ABC_invT;

                    for (int x = xmin; x <= xmax; ++x) {
                        for (int y = ymin; y <= ymax; ++y) {
                            vec3 bc_screen = ABC_invT * vec3{(double)x, (double)y, 1.};
                            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
                                continue;

                            // perspective-correct barycentric
                            vec3 bc_clip = {
                                bc_screen.x / face.clip[0].w,
                                bc_screen.y / face.clip[1].w,
                                bc_screen.z / face.clip[2].w
                            };
                            double sum = bc_clip.x + bc_clip.y + bc_clip.z;
                            if (sum == 0.0) continue;
                            bc_clip = bc_clip / sum;

                            // 深度插值（使用 NDC z）
                            double z = bc_screen * vec3{
                                face.ndc[0].z,
                                face.ndc[1].z,
                                face.ndc[2].z
                            };

                            int idx = x + y * width;
                            if (z <= zbuffer[idx]) continue;

                            auto [discard, color] = shader.fragment(bc_clip);
                            if (discard) continue;

                            // ⚠️ 每個 pixel 僅由一個 tile／thread 負責，無需 lock
                            zbuffer[idx] = z;
                            framebuffer.set(x, y, color);
                        }
                    }
                }
            }
        });
    }

    for (auto &t : workers) {
        t.join();
    }
}

int main(int argc, char** argv) {
    double start = CycleTimer::current_seconds();

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_threads> <--mode [face|tile]> model1.obj [model2.obj ...]\n";
        return 1;
    }

    /* ---------- parse threads ---------- */
    GLOBAL_THREAD_COUNT = std::max(1, std::atoi(argv[1]));

    /* ---------- parse mode ---------- */
    if (std::string(argv[2]) != "--mode") {
        std::cerr << "Expected --mode\n";
        return 1;
    }

    enum class RenderMode { FACE, TILE };
    RenderMode mode;

    std::string mode_str = argv[3];
    if (mode_str == "face") 
        mode = RenderMode::FACE;
    else if (mode_str == "tile") 
        mode = RenderMode::TILE;
    else {
        std::cerr << "Invalid mode: " << mode_str << "\n";
        return 1;
    }

    /* ---------- camera / framebuffer ---------- */
    constexpr int width  = 800;
    constexpr int height = 800;
    constexpr vec3  light{1, 1, 1};
    constexpr vec3    eye{-1, 0, 2};
    constexpr vec3 center{0, 0, 0};
    constexpr vec3     up{0, 1, 0};

    lookat(eye, center, up);
    init_perspective(norm(eye - center));
    init_viewport(width/16, height/16, width*7/8, height*7/8);
    init_zbuffer(width, height);

    TGAImage framebuffer(width, height, TGAImage::RGB,
                          {177, 195, 209, 255});

    /* ---------- render models ---------- */
    const int tile_size = 16;

    for (int m = 4; m < argc; ++m) {
        Model model(argv[m]);

        if (mode == RenderMode::FACE)
            render_model_face_parallel(model, light, framebuffer, GLOBAL_THREAD_COUNT);
        else 
            render_model_tiled(model, light, framebuffer, tile_size, GLOBAL_THREAD_COUNT);
    }

    double process_time = CycleTimer::current_seconds() - start;

    framebuffer.write_tga_file("framebuffer.tga");
    std::cout << "Render time: "
              << process_time << " seconds\n";

    return 0;
}