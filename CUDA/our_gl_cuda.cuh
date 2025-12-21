#pragma once
#include "geometry.h"
#include "tgaimage.h"
#include <vector>

// Forward declaration for CUDA types
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

// Structure to hold triangle data for CUDA
struct CudaTriangle {
    vec4 clip[3];        // Clip space coordinates
    vec2 screen[3];      // Screen coordinates  
    vec2 varying_uv[3];  // UV coordinates
    vec4 varying_nrm[3]; // Normals
    vec4 view_tri[3];    // View space coordinates
    double abc_inv_transpose[3][3]; // Precomputed inverse transpose matrix for barycentric coords
    double abc_det;      // Precomputed determinant for backface culling
    int bbmin_x, bbmin_y, bbmax_x, bbmax_y; // Bounding box for tile culling
};

// Structure to hold texture data
struct CudaTexture {
    unsigned char* data;
    int width;
    int height;
    int bytespp;
};

// Host-callable CUDA functions
void cuda_init();
void cuda_cleanup();

// 優化的 CUDA 渲染器：以 face 為單位，GPU 上保持全局 framebuffer 和 zbuffer
class CudaRenderer {
public:
    CudaRenderer(int width, int height);
    ~CudaRenderer();
    
    void upload_textures(const TGAImage& diffuse, const TGAImage& normal, const TGAImage& specular);
    
    void init_framebuffer(const TGAColor& bg_color);
    
    void render_triangle(
        const CudaTriangle& triangle,
        const vec4& light_dir,
        const mat<4,4>& viewport
    );
    
    void download_framebuffer(TGAImage& framebuffer);

private:
    int width_, height_;
    
    // Global buffers on the GPU
    unsigned char* d_framebuffer_;
    double* d_zbuffer_;
    
    // Texture data on the GPU
    unsigned char* d_diffuse_data_;
    unsigned char* d_normal_data_;
    unsigned char* d_specular_data_;
    int diffuse_width_, diffuse_height_;
    int normal_width_, normal_height_;
    int specular_width_, specular_height_;
    
    bool textures_uploaded_;
};

// face-level streaming CUDA
class CudaStreamRenderer {
public:
    CudaStreamRenderer(int width, int height, int num_streams);
    ~CudaStreamRenderer();
    
    void upload_textures(const TGAImage& diffuse, const TGAImage& normal, const TGAImage& specular);

    void init_framebuffer(const TGAColor& bg_color);
    
    void render_triangles_batch(
        const std::vector<CudaTriangle>& triangles,
        const vec4& light_dir,
        const mat<4,4>& viewport
    );
    

    void download_framebuffer(TGAImage& framebuffer);
    
private:
    int width_, height_;
    int num_streams_;
    
    // CUDA streams
    cudaStream_t* streams_;
    
    // Global buffers on the GPU
    unsigned char* d_framebuffer_;
    double* d_zbuffer_;
    
    // Texture data on the GPU
    unsigned char* d_diffuse_data_;
    unsigned char* d_normal_data_;
    unsigned char* d_specular_data_;
    int diffuse_width_, diffuse_height_;
    int normal_width_, normal_height_;
    int specular_width_, specular_height_;
    
    bool textures_uploaded_;
};

// Tile-based CUDA
class CudaTileRenderer {
public:
    CudaTileRenderer(int width, int height, int tile_size = 16);
    ~CudaTileRenderer();
    
    void upload_textures(const TGAImage& diffuse, const TGAImage& normal, const TGAImage& specular);
    void init_framebuffer(const TGAColor& bg_color);
    void render_triangles_batch(const std::vector<CudaTriangle>& triangles, const vec4& light_dir, const mat<4,4>& viewport);
    void download_framebuffer(TGAImage& framebuffer);
    
private:
    int width_, height_;
    int tile_size_;
    int tiles_x_, tiles_y_;
    int total_tiles_;
    
    unsigned char* d_framebuffer_;
    double* d_zbuffer_;
    
    unsigned char* d_diffuse_data_;
    unsigned char* d_normal_data_;
    unsigned char* d_specular_data_;
    int diffuse_width_, diffuse_height_;
    int normal_width_, normal_height_;
    int specular_width_, specular_height_;
    
    bool textures_uploaded_;
};