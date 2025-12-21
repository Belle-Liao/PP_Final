#include "our_gl.h"
#include "our_gl_cuda.cuh"
#include "model.h"
#include "cycle_timer.h"
#include <iomanip>
#include <iostream>

extern mat<4,4> ModelView, Perspective, Viewport; // "OpenGL" state matrices
extern std::vector<double> zbuffer;                // the depth buffer

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
        varying_uv[vert]  = model.uv(face, vert);                        // read the uv from the model
        varying_nrm[vert] = ModelView.invert_transpose() * model.normal(face, vert); // transform the normal to view coordinates
        vec4 gl_Position = ModelView * model.vert(face, vert);
        tri[vert] = gl_Position;
        return Perspective * gl_Position; // transform vertex to clip coordinates
    }

    virtual std::pair<bool,TGAColor> fragment(const vec3 bc) const {
        vec2 uv = varying_uv[0]*bc.x + varying_uv[1]*bc.y + varying_uv[2]*bc.z;     // interpolate uv for the current pixel
        vec4 n = normalized(varying_nrm[0]*bc.x + varying_nrm[1]*bc.y + varying_nrm[2]*bc.z); // interpolate the normal
        
        // Simple diffuse + specular lighting
        double nl = std::max(0., n*l);                               // light-normal dot-product
        vec4 r = n*nl*2. - l; r.w=0;                                 // reflected light direction
        
        // Get specular intensity from normal map's red channel
        TGAColor spec_color = model.normal_map().get(uv.x * model.normal_map().width(), uv.y * model.normal_map().height());
        double specular = std::pow(std::max(0., r.z), spec_color.bgra[0] / 1.0); // specular coefficient
        
        // Get color from diffuse map
        TGAColor c = model.diffuse().get(uv.x * model.diffuse().width(), uv.y * model.diffuse().height());
        for (int i : {0,1,2})
            c[i] = std::min<int>(c[i]*(.4 + 1.*nl + specular), 255); // multiply the color by the intensity

        return {false, c};
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                << " [stream|tile] model.obj [model2.obj ...]\n";
        return 1;
    }

    enum class RenderMode { STREAM=1, TILE=2 };
    RenderMode render_mode;
    std::string mode_arg = argv[1];
    if (mode_arg == "stream") render_mode = RenderMode::STREAM;
    else if (mode_arg == "tile")   render_mode = RenderMode::TILE;
    else {
        std::cerr << "Unknown render mode: " << mode_arg << "\n";
        return 1;
    }

    constexpr int width  = 800;                 // output image size
    constexpr int height = 800;
    constexpr vec3 light{0, 0, 1};              // light source
    constexpr vec3    eye{-1, 0, 2}; // camera position
    constexpr vec3 center{ 0, 0, 0}; // camera direction
    constexpr vec3     up{ 0, 1, 0}; // camera up vector

    double total_start = CycleTimer::current_seconds();
    
    // CUDA initialization
    cuda_init();
    
    lookat(eye, center, up);                                   // build the ModelView   matrix
    init_perspective(norm(eye-center));                        // build the Perspective matrix
    init_viewport(width/16, height/16, width*7/8, height*7/8); // build the Viewport    matrix
    init_zbuffer(width, height);
    TGAImage framebuffer(width, height, TGAImage::RGB, {177, 195, 209, 255});

    CudaRenderer* cuda_renderer = new CudaRenderer(width, height);
    CudaStreamRenderer* stream_renderer = new CudaStreamRenderer(width, height, 8);
    CudaTileRenderer* tile_renderer = new CudaTileRenderer(width, height, 8);
    
    int obj_start = 2;
    for (int m=obj_start; m<argc; m++) {
        Model model(argv[m]);
        PhongShader shader(light, model);

        // Start timing for rendering
        double render_start = CycleTimer::current_seconds();

        // Reset framebuffer and render based on mode
        if (RenderMode::STREAM) {
            stream_renderer->init_framebuffer({177, 195, 209, 255});
            stream_renderer->render_triangles_batch(all_triangles, shader.l, Viewport);
            stream_renderer->download_framebuffer(framebuffer);
        } else if (render_mode == RenderMode::TILE) {
            tile_renderer->download_framebuffer(framebuffer);
        }

        double render_time = CycleTimer::current_seconds() - render_start;
        std::cout << "Render time: " << std::setprecision(4) << render_time << " sec" << std::endl;
    }

    delete cuda_renderer;
    delete stream_renderer;
    delete tile_renderer;

    double before_write_time = CycleTimer::current_seconds() - total_start;
    
    std::cout << "Total time: " << std::setprecision(4) << before_write_time << " sec" << std::endl;

    framebuffer.write_tga_file("framebuffer.tga");
    cuda_cleanup();
    
    return 0;
}
