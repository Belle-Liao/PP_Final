#include "our_gl.h"
#include "model.h"
#include "cycle_timer.h"

extern mat<4,4> ModelView, Perspective; // "OpenGL" state matrices and
extern std::vector<double> zbuffer;     // the depth buffer

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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                << " --mode [face|tile] obj/model.obj [obj2 ...]\n";
        return 1;
    }   

    enum class ParallelMode { FACE, TILE };
    ParallelMode mode;

    if (std::string(argv[1]) == "--mode") {
        std::string v = argv[2];
        if      (v == "face") mode = ParallelMode::FACE;
        else if (v == "tile") mode = ParallelMode::TILE;
        else {
            std::cerr << "Invalid mode: " << v << "\n";
            return 1;
        }
    } 
    else {
        std::cerr << "Expected --mode\n";
        return 1;
    }

    int obj_start = 3;

    double t2 = CycleTimer::current_seconds();
    constexpr int width  = 800;      // output image size
    constexpr int height = 800;
    constexpr vec3  light{ 1, 1, 1}; // light source
    constexpr vec3    eye{-1, 0, 2}; // camera position
    constexpr vec3 center{ 0, 0, 0}; // camera direction
    constexpr vec3     up{ 0, 1, 0}; // camera up vector

    lookat(eye, center, up);                                   // build the ModelView   matrix
    init_perspective(norm(eye-center));                        // build the Perspective matrix
    init_viewport(width/16, height/16, width*7/8, height*7/8); // build the Viewport    matrix
    init_zbuffer(width, height);
    TGAImage framebuffer(width, height, TGAImage::RGB, {177, 195, 209, 255});

    for (int m = obj_start; m < argc; m++) {                    // iterate through all input objects
        Model model(argv[m]);                       // load the data
        PhongShader shader(light, model);
        
        if (mode == ParallelMode::FACE) {
            #pragma omp parallel for schedule(dynamic)
            for (int f = 0; f < model.nfaces(); f++) {
                Triangle clip = { shader.vertex(f, 0),  // assemble the primitive
                                  shader.vertex(f, 1),
                                  shader.vertex(f, 2) };
                rasterize_face(clip, shader, framebuffer);   // rasterize the primitive
            }
        } 
        else if (mode == ParallelMode::TILE) {
            #pragma omp parallel for schedule(dynamic)
            for (int f = 0; f < model.nfaces(); f++) {
                Triangle clip = { shader.vertex(f, 0),  // assemble the primitive
                                  shader.vertex(f, 1),
                                  shader.vertex(f, 2) };
                rasterize_tile(clip, shader, framebuffer);   // rasterize the primitive
            }
        }
    }

    double t3 = CycleTimer::current_seconds();
    std::cout << "Parallel OpenMP: "
              << (t3-t2)
              << " ms\n";
    framebuffer.write_tga_file("framebuffer.tga");
    return 0;
}
