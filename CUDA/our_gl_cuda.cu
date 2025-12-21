#include "our_gl_cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device helper functions
__device__ double dot_device(const vec4& a, const vec4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ double dot_device(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ double norm_device(const vec4& v) {
    return sqrt(dot_device(v, v));
}

__device__ double norm_device(const vec3& v) {
    return sqrt(dot_device(v, v));
}

__device__ vec4 normalized_device(const vec4& v) {
    double n = norm_device(v);
    if (n < 1e-8) return v;
    return {v.x/n, v.y/n, v.z/n, v.w/n};
}

__device__ vec3 normalized_device(const vec3& v) {
    double n = norm_device(v);
    if (n < 1e-8) return v;
    return {v.x/n, v.y/n, v.z/n};
}

__device__ void sample2D_device(const unsigned char* img_data, int width, int height, int bytespp,
                                const vec2& uvf, unsigned char* out) {
    int x = (int)(uvf.x * width);
    int y = (int)(uvf.y * height);
    if (x < 0) x = 0;
    if (x >= width) x = width - 1;
    if (y < 0) y = 0;
    if (y >= height) y = height - 1;
    
    int idx = (x + y * width) * bytespp;
    for (int i = 0; i < bytespp; i++) {
        out[i] = img_data[idx + i];
    }
}

void cuda_init() {
    CUDA_CHECK(cudaSetDevice(0));
    
    // 最輕量級預熱：只觸發最關鍵的初始化
    // 分配極小記憶體來觸發記憶體管理器初始化
    void* dummy_ptr;
    CUDA_CHECK(cudaMalloc(&dummy_ptr, 1));
    CUDA_CHECK(cudaFree(dummy_ptr));
    
    // 執行一次同步確保context完全建立
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_cleanup() {
    CUDA_CHECK(cudaDeviceReset());
}

// ============================================================================
// 優化的 CUDA 渲染器：使用 2D grid，每個 thread 處理一個像素
// ============================================================================

// 優化的kernel：使用2D grid，每個thread處理一個像素
__global__ void render_triangle_kernel_optimized(
    CudaTriangle triangle,
    CudaTexture diffuse_tex,
    CudaTexture normal_tex,
    CudaTexture specular_tex,
    vec4 light_dir,
    unsigned char* framebuffer_data,
    double* zbuffer,
    int fb_width, int fb_height,
    int bbminx, int bbmaxx,
    int bbminy, int bbmaxy,
    mat<3,3> abc_inv_transpose
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + bbminx;
    int y = blockIdx.y * blockDim.y + threadIdx.y + bbminy;
    
    if (x > bbmaxx || x >= fb_width || y > bbmaxy || y >= fb_height) return;
    
    // Compute barycentric coordinates
    vec3 bc_screen;
    bc_screen.x = abc_inv_transpose[0][0] * x + abc_inv_transpose[0][1] * y + abc_inv_transpose[0][2];
    bc_screen.y = abc_inv_transpose[1][0] * x + abc_inv_transpose[1][1] * y + abc_inv_transpose[1][2];
    bc_screen.z = abc_inv_transpose[2][0] * x + abc_inv_transpose[2][1] * y + abc_inv_transpose[2][2];
    
    if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) return;
    
    // Perspective-correct barycentric coordinates
    vec3 bc_clip;
    bc_clip.x = bc_screen.x / triangle.clip[0].w;
    bc_clip.y = bc_screen.y / triangle.clip[1].w;
    bc_clip.z = bc_screen.z / triangle.clip[2].w;
    double bc_sum = bc_clip.x + bc_clip.y + bc_clip.z;
    bc_clip.x /= bc_sum;
    bc_clip.y /= bc_sum;
    bc_clip.z /= bc_sum;
    
    // Interpolate depth (NDC z)
    vec4 ndc[3];
    for (int i = 0; i < 3; i++) {
        ndc[i] = {triangle.clip[i].x/triangle.clip[i].w, 
                  triangle.clip[i].y/triangle.clip[i].w, 
                  triangle.clip[i].z/triangle.clip[i].w, 1.0};
    }
    
    double z = bc_screen.x * ndc[0].z + bc_screen.y * ndc[1].z + bc_screen.z * ndc[2].z;
    
    int zbuf_idx = x + y * fb_width;
    
    // Z-buffer test with atomic operation
    unsigned long long* zbuf_addr = (unsigned long long*)&zbuffer[zbuf_idx];
    unsigned long long assumed;
    unsigned long long old_val = __double_as_longlong(zbuffer[zbuf_idx]);
    
    do {
        assumed = old_val;
        if (__longlong_as_double(assumed) >= z) return;
        old_val = atomicCAS(zbuf_addr, assumed, __double_as_longlong(z));
    } while (assumed != old_val);
    
    // Fragment shader computations
    // Interpolate UV
    vec2 uv;
    uv.x = triangle.varying_uv[0].x * bc_clip.x + triangle.varying_uv[1].x * bc_clip.y + triangle.varying_uv[2].x * bc_clip.z;
    uv.y = triangle.varying_uv[0].y * bc_clip.x + triangle.varying_uv[1].y * bc_clip.y + triangle.varying_uv[2].y * bc_clip.z;
    
    // Compute tangent and bitangent for normal mapping
    vec4 E0 = {triangle.view_tri[1].x - triangle.view_tri[0].x, 
               triangle.view_tri[1].y - triangle.view_tri[0].y, 
               triangle.view_tri[1].z - triangle.view_tri[0].z, 
               triangle.view_tri[1].w - triangle.view_tri[0].w};
    vec4 E1 = {triangle.view_tri[2].x - triangle.view_tri[0].x, 
               triangle.view_tri[2].y - triangle.view_tri[0].y,
               triangle.view_tri[2].z - triangle.view_tri[0].z, 
               triangle.view_tri[2].w - triangle.view_tri[0].w};
    
    vec2 dUV0 = {triangle.varying_uv[1].x - triangle.varying_uv[0].x, 
                 triangle.varying_uv[1].y - triangle.varying_uv[0].y};
    vec2 dUV1 = {triangle.varying_uv[2].x - triangle.varying_uv[0].x, 
                 triangle.varying_uv[2].y - triangle.varying_uv[0].y};
    
    double det_uv = dUV0.x * dUV1.y - dUV0.y * dUV1.x;
    if (fabs(det_uv) < 1e-8) det_uv = 1e-8;
    
    vec4 tangent = {
        (dUV1.y * E0.x - dUV0.y * E1.x) / det_uv,
        (dUV1.y * E0.y - dUV0.y * E1.y) / det_uv,
        (dUV1.y * E0.z - dUV0.y * E1.z) / det_uv,
        (dUV1.y * E0.w - dUV0.y * E1.w) / det_uv
    };
    
    vec4 bitangent = {
        (-dUV1.x * E0.x + dUV0.x * E1.x) / det_uv,
        (-dUV1.x * E0.y + dUV0.x * E1.y) / det_uv,
        (-dUV1.x * E0.z + dUV0.x * E1.z) / det_uv,
        (-dUV1.x * E0.w + dUV0.x * E1.w) / det_uv
    };
    
    tangent = normalized_device(tangent);
    bitangent = normalized_device(bitangent);
    
    // Interpolate normal
    vec4 interp_nrm;
    interp_nrm.x = triangle.varying_nrm[0].x * bc_clip.x + triangle.varying_nrm[1].x * bc_clip.y + triangle.varying_nrm[2].x * bc_clip.z;
    interp_nrm.y = triangle.varying_nrm[0].y * bc_clip.x + triangle.varying_nrm[1].y * bc_clip.y + triangle.varying_nrm[2].y * bc_clip.z;
    interp_nrm.z = triangle.varying_nrm[0].z * bc_clip.x + triangle.varying_nrm[1].z * bc_clip.y + triangle.varying_nrm[2].z * bc_clip.z;
    interp_nrm.w = triangle.varying_nrm[0].w * bc_clip.x + triangle.varying_nrm[1].w * bc_clip.y + triangle.varying_nrm[2].w * bc_clip.z;
    interp_nrm = normalized_device(interp_nrm);
    
    // Sample normal map
    unsigned char normal_sample[4];
    sample2D_device(normal_tex.data, normal_tex.width, normal_tex.height, normal_tex.bytespp, uv, normal_sample);
    
    // Transform normal from tangent space to view space
    vec4 n_tangent = {
        (normal_sample[2] / 255.0) * 2.0 - 1.0,
        (normal_sample[1] / 255.0) * 2.0 - 1.0,
        (normal_sample[0] / 255.0) * 2.0 - 1.0,
        0.0
    };
    
    vec4 n;
    n.x = tangent.x * n_tangent.x + bitangent.x * n_tangent.y + interp_nrm.x * n_tangent.z;
    n.y = tangent.y * n_tangent.x + bitangent.y * n_tangent.y + interp_nrm.y * n_tangent.z;
    n.z = tangent.z * n_tangent.x + bitangent.z * n_tangent.y + interp_nrm.z * n_tangent.z;
    n.w = 0.0;
    n = normalized_device(n);
    
    // Lighting calculations
    double nl = dot_device(n, light_dir);
    
    vec4 r;
    r.x = n.x * nl * 2.0 - light_dir.x;
    r.y = n.y * nl * 2.0 - light_dir.y;
    r.z = n.z * nl * 2.0 - light_dir.z;
    r.w = 0.0;
    r = normalized_device(r);
    
    double ambient = 0.4;
    double diffuse = 1.0 * fmax(0.0, nl);
    
    unsigned char spec_sample[4];
    sample2D_device(specular_tex.data, specular_tex.width, specular_tex.height, specular_tex.bytespp, uv, spec_sample);
    double specular = (0.5 + 2.0 * spec_sample[0] / 255.0) * pow(fmax(r.z, 0.0), 35.0);
    
    // Sample diffuse texture
    unsigned char diff_sample[4];
    sample2D_device(diffuse_tex.data, diffuse_tex.width, diffuse_tex.height, diffuse_tex.bytespp, uv, diff_sample);
    
    // Apply lighting and write to framebuffer
    int fb_idx = (x + y * fb_width) * 3; // RGB format
    double lighting = ambient + diffuse + specular;
    framebuffer_data[fb_idx + 2] = (unsigned char)min(255.0, diff_sample[2] * lighting); // R
    framebuffer_data[fb_idx + 1] = (unsigned char)min(255.0, diff_sample[1] * lighting); // G
    framebuffer_data[fb_idx + 0] = (unsigned char)min(255.0, diff_sample[0] * lighting); // B
}

// ============================================================================
// CudaRenderer 實現：優化的 face-level 渲染器
// ============================================================================

// 優化的kernel：使用2D grid，每個thread處理一個像素
__global__ void render_triangle_kernel_optimized(
    CudaTriangle triangle,
    CudaTexture diffuse_tex,
    CudaTexture normal_tex,
    CudaTexture specular_tex,
    vec4 light_dir,
    unsigned char* framebuffer_data,
    double* zbuffer,
    int fb_width, int fb_height,
    mat<3,3> abc_inv_transpose,
    int bbminx, int bbmaxx,
    int bbminy, int bbmaxy
) {
    // 使用2D grid計算當前像素位置
    int x = blockIdx.x * blockDim.x + threadIdx.x + bbminx;
    int y = blockIdx.y * blockDim.y + threadIdx.y + bbminy;
    
    if (x > bbmaxx || x >= fb_width || y > bbmaxy || y >= fb_height) return;
        
    // Compute barycentric coordinates
    vec3 bc_screen;
    bc_screen.x = abc_inv_transpose[0][0] * x + abc_inv_transpose[0][1] * y + abc_inv_transpose[0][2];
    bc_screen.y = abc_inv_transpose[1][0] * x + abc_inv_transpose[1][1] * y + abc_inv_transpose[1][2];
    bc_screen.z = abc_inv_transpose[2][0] * x + abc_inv_transpose[2][1] * y + abc_inv_transpose[2][2];
    
    if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) return;
    
    // Perspective-correct barycentric coordinates
    vec3 bc_clip;
        bc_clip.x = bc_screen.x / triangle.clip[0].w;
        bc_clip.y = bc_screen.y / triangle.clip[1].w;
        bc_clip.z = bc_screen.z / triangle.clip[2].w;
        double bc_sum = bc_clip.x + bc_clip.y + bc_clip.z;
        bc_clip.x /= bc_sum;
        bc_clip.y /= bc_sum;
        bc_clip.z /= bc_sum;
        
        // Interpolate depth
        vec4 ndc[3];
        for (int i = 0; i < 3; i++) {
            ndc[i] = {triangle.clip[i].x/triangle.clip[i].w, 
                      triangle.clip[i].y/triangle.clip[i].w, 
                      triangle.clip[i].z/triangle.clip[i].w, 1.0};
        }
        
        double z = bc_screen.x * ndc[0].z + bc_screen.y * ndc[1].z + bc_screen.z * ndc[2].z;
        
        int zbuf_idx = x + y * fb_width;
        
        // Z-buffer test with atomic operation
        unsigned long long* zbuf_addr = (unsigned long long*)&zbuffer[zbuf_idx];
        unsigned long long assumed;
        unsigned long long old_val = __double_as_longlong(zbuffer[zbuf_idx]);
        
        do {
            assumed = old_val;
            if (__longlong_as_double(assumed) >= z) break;
            old_val = atomicCAS(zbuf_addr, assumed, __double_as_longlong(z));
        } while (assumed != old_val);
        
        if (__longlong_as_double(assumed) >= z) return;
        
        // Fragment shader computations
        vec2 uv;
        uv.x = triangle.varying_uv[0].x * bc_clip.x + triangle.varying_uv[1].x * bc_clip.y + triangle.varying_uv[2].x * bc_clip.z;
        uv.y = triangle.varying_uv[0].y * bc_clip.x + triangle.varying_uv[1].y * bc_clip.y + triangle.varying_uv[2].y * bc_clip.z;
        
        // Compute tangent and bitangent
        vec4 E0 = {triangle.view_tri[1].x - triangle.view_tri[0].x, 
                   triangle.view_tri[1].y - triangle.view_tri[0].y, 
                   triangle.view_tri[1].z - triangle.view_tri[0].z, 
                   triangle.view_tri[1].w - triangle.view_tri[0].w};
        vec4 E1 = {triangle.view_tri[2].x - triangle.view_tri[0].x, 
                   triangle.view_tri[2].y - triangle.view_tri[0].y,
                   triangle.view_tri[2].z - triangle.view_tri[0].z, 
                   triangle.view_tri[2].w - triangle.view_tri[0].w};
        
        vec2 dUV0 = {triangle.varying_uv[1].x - triangle.varying_uv[0].x, 
                     triangle.varying_uv[1].y - triangle.varying_uv[0].y};
        vec2 dUV1 = {triangle.varying_uv[2].x - triangle.varying_uv[0].x, 
                     triangle.varying_uv[2].y - triangle.varying_uv[0].y};
        
        double det_uv = dUV0.x * dUV1.y - dUV0.y * dUV1.x;
        if (fabs(det_uv) < 1e-8) det_uv = 1e-8;
        
        vec4 tangent = {
            (dUV1.y * E0.x - dUV0.y * E1.x) / det_uv,
            (dUV1.y * E0.y - dUV0.y * E1.y) / det_uv,
            (dUV1.y * E0.z - dUV0.y * E1.z) / det_uv,
            (dUV1.y * E0.w - dUV0.y * E1.w) / det_uv
        };
        
        vec4 bitangent = {
            (-dUV1.x * E0.x + dUV0.x * E1.x) / det_uv,
            (-dUV1.x * E0.y + dUV0.x * E1.y) / det_uv,
            (-dUV1.x * E0.z + dUV0.x * E1.z) / det_uv,
            (-dUV1.x * E0.w + dUV0.x * E1.w) / det_uv
        };
        
        tangent = normalized_device(tangent);
        bitangent = normalized_device(bitangent);
        
        // Interpolate normal
        vec4 interp_nrm;
        interp_nrm.x = triangle.varying_nrm[0].x * bc_clip.x + triangle.varying_nrm[1].x * bc_clip.y + triangle.varying_nrm[2].x * bc_clip.z;
        interp_nrm.y = triangle.varying_nrm[0].y * bc_clip.x + triangle.varying_nrm[1].y * bc_clip.y + triangle.varying_nrm[2].y * bc_clip.z;
        interp_nrm.z = triangle.varying_nrm[0].z * bc_clip.x + triangle.varying_nrm[1].z * bc_clip.y + triangle.varying_nrm[2].z * bc_clip.z;
        interp_nrm.w = triangle.varying_nrm[0].w * bc_clip.x + triangle.varying_nrm[1].w * bc_clip.y + triangle.varying_nrm[2].w * bc_clip.z;
        interp_nrm = normalized_device(interp_nrm);
        
        // Sample normal map
        unsigned char normal_sample[4];
        sample2D_device(normal_tex.data, normal_tex.width, normal_tex.height, normal_tex.bytespp, uv, normal_sample);
        
        // Transform normal from tangent space to view space
        vec4 n_tangent = {
            (normal_sample[2] / 255.0) * 2.0 - 1.0,
            (normal_sample[1] / 255.0) * 2.0 - 1.0,
            (normal_sample[0] / 255.0) * 2.0 - 1.0,
            0.0
        };
        
        vec4 n;
        n.x = tangent.x * n_tangent.x + bitangent.x * n_tangent.y + interp_nrm.x * n_tangent.z;
        n.y = tangent.y * n_tangent.x + bitangent.y * n_tangent.y + interp_nrm.y * n_tangent.z;
        n.z = tangent.z * n_tangent.x + bitangent.z * n_tangent.y + interp_nrm.z * n_tangent.z;
        n.w = 0.0;
        n = normalized_device(n);
        
        // Lighting calculations
        double nl = dot_device(n, light_dir);
        
        vec4 r;
        r.x = n.x * nl * 2.0 - light_dir.x;
        r.y = n.y * nl * 2.0 - light_dir.y;
        r.z = n.z * nl * 2.0 - light_dir.z;
        r.w = 0.0;
        r = normalized_device(r);
        
        double ambient = 0.4;
        double diffuse = 1.0 * fmax(0.0, nl);
        
        unsigned char spec_sample[4];
        sample2D_device(specular_tex.data, specular_tex.width, specular_tex.height, specular_tex.bytespp, uv, spec_sample);
        double specular = (0.5 + 2.0 * spec_sample[0] / 255.0) * pow(fmax(r.z, 0.0), 35.0);
        
        // Sample diffuse texture
        unsigned char diff_sample[4];
        sample2D_device(diffuse_tex.data, diffuse_tex.width, diffuse_tex.height, diffuse_tex.bytespp, uv, diff_sample);
        
    // Apply lighting and write to framebuffer
    int fb_idx = (x + y * fb_width) * 3;
    double lighting = ambient + diffuse + specular;
    framebuffer_data[fb_idx + 2] = (unsigned char)min(255.0, diff_sample[2] * lighting);
    framebuffer_data[fb_idx + 1] = (unsigned char)min(255.0, diff_sample[1] * lighting);
    framebuffer_data[fb_idx + 0] = (unsigned char)min(255.0, diff_sample[0] * lighting);
}

// CudaRenderer implementation
CudaRenderer::CudaRenderer(int width, int height) 
    : width_(width), height_(height), textures_uploaded_(false),
      d_framebuffer_(nullptr), d_zbuffer_(nullptr),
      d_diffuse_data_(nullptr), d_normal_data_(nullptr), d_specular_data_(nullptr) {
    
    // Allocate GPU memory for framebuffer and zbuffer
    size_t fb_size = width * height * 3;
    size_t zbuf_size = width * height * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_framebuffer_, fb_size));
    CUDA_CHECK(cudaMalloc(&d_zbuffer_, zbuf_size));
    
    // Initialize zbuffer to -1000
    std::vector<double> init_zbuf(width * height, -1000.0);
    CUDA_CHECK(cudaMemcpy(d_zbuffer_, init_zbuf.data(), zbuf_size, cudaMemcpyHostToDevice));
}

CudaRenderer::~CudaRenderer() {
    if (d_framebuffer_) CUDA_CHECK(cudaFree(d_framebuffer_));
    if (d_zbuffer_) CUDA_CHECK(cudaFree(d_zbuffer_));
    if (d_diffuse_data_) CUDA_CHECK(cudaFree(d_diffuse_data_));
    if (d_normal_data_) CUDA_CHECK(cudaFree(d_normal_data_));
    if (d_specular_data_) CUDA_CHECK(cudaFree(d_specular_data_));
}

void CudaRenderer::upload_textures(const TGAImage& diffuse, const TGAImage& normal, const TGAImage& specular) {
    // Free old textures if they exist
    if (d_diffuse_data_) CUDA_CHECK(cudaFree(d_diffuse_data_));
    if (d_normal_data_) CUDA_CHECK(cudaFree(d_normal_data_));
    if (d_specular_data_) CUDA_CHECK(cudaFree(d_specular_data_));
    
    diffuse_width_ = diffuse.width();
    diffuse_height_ = diffuse.height();
    normal_width_ = normal.width();
    normal_height_ = normal.height();
    specular_width_ = specular.width();
    specular_height_ = specular.height();
    
    // Prepare texture data on host
    std::vector<unsigned char> diffuse_data(diffuse_width_ * diffuse_height_ * 3);
    std::vector<unsigned char> normal_data(normal_width_ * normal_height_ * 3);
    std::vector<unsigned char> specular_data(specular_width_ * specular_height_ * 1);
    
    for (int y = 0; y < diffuse_height_; y++) {
        for (int x = 0; x < diffuse_width_; x++) {
            TGAColor c = diffuse.get(x, y);
            int idx = (x + y * diffuse_width_) * 3;
            diffuse_data[idx] = c.bgra[0];
            diffuse_data[idx + 1] = c.bgra[1];
            diffuse_data[idx + 2] = c.bgra[2];
        }
    }
    
    for (int y = 0; y < normal_height_; y++) {
        for (int x = 0; x < normal_width_; x++) {
            TGAColor c = normal.get(x, y);
            int idx = (x + y * normal_width_) * 3;
            normal_data[idx] = c.bgra[0];
            normal_data[idx + 1] = c.bgra[1];
            normal_data[idx + 2] = c.bgra[2];
        }
    }
    
    for (int y = 0; y < specular_height_; y++) {
        for (int x = 0; x < specular_width_; x++) {
            TGAColor c = specular.get(x, y);
            int idx = x + y * specular_width_;
            specular_data[idx] = c.bgra[0];
        }
    }
    
    // Upload to GPU
    size_t diffuse_size = diffuse_width_ * diffuse_height_ * 3;
    size_t normal_size = normal_width_ * normal_height_ * 3;
    size_t specular_size = specular_width_ * specular_height_ * 1;
    
    CUDA_CHECK(cudaMalloc(&d_diffuse_data_, diffuse_size));
    CUDA_CHECK(cudaMalloc(&d_normal_data_, normal_size));
    CUDA_CHECK(cudaMalloc(&d_specular_data_, specular_size));
    
    CUDA_CHECK(cudaMemcpy(d_diffuse_data_, diffuse_data.data(), diffuse_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normal_data_, normal_data.data(), normal_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_specular_data_, specular_data.data(), specular_size, cudaMemcpyHostToDevice));
    
    textures_uploaded_ = true;
}

void CudaRenderer::init_framebuffer(const TGAColor& bg_color) {
    std::vector<unsigned char> fb_data(width_ * height_ * 3);
    for (int i = 0; i < width_ * height_; i++) {
        fb_data[i * 3 + 0] = bg_color.bgra[0];
        fb_data[i * 3 + 1] = bg_color.bgra[1];
        fb_data[i * 3 + 2] = bg_color.bgra[2];
    }
    CUDA_CHECK(cudaMemcpy(d_framebuffer_, fb_data.data(), width_ * height_ * 3, cudaMemcpyHostToDevice));
}

void CudaRenderer::render_triangle(
    const CudaTriangle& triangle,
    const vec4& light_dir,
    const mat<4,4>& viewport
) {
    if (!textures_uploaded_) {
        fprintf(stderr, "Error: Textures not uploaded!\n");
        return;
    }
    
    // Compute bounding box
    int bbminx = std::max<int>(std::min({triangle.screen[0].x, triangle.screen[1].x, triangle.screen[2].x}), 0);
    int bbmaxx = std::min<int>(std::max({triangle.screen[0].x, triangle.screen[1].x, triangle.screen[2].x}), width_ - 1);
    int bbminy = std::max<int>(std::min({triangle.screen[0].y, triangle.screen[1].y, triangle.screen[2].y}), 0);
    int bbmaxy = std::min<int>(std::max({triangle.screen[0].y, triangle.screen[1].y, triangle.screen[2].y}), height_ - 1);
    
    if (bbminx > bbmaxx || bbminy > bbmaxy) return;
    
    // Compute ABC matrix
    mat<3,3> ABC = {{
        {triangle.screen[0].x, triangle.screen[0].y, 1.},
        {triangle.screen[1].x, triangle.screen[1].y, 1.},
        {triangle.screen[2].x, triangle.screen[2].y, 1.}
    }};
    
    double abc_det = ABC.det();
    if (abc_det < 1) return;
    
    mat<3,3> abc_inv_transpose = ABC.invert_transpose();
    
    // Create texture structs
    CudaTexture diffuse_tex = {d_diffuse_data_, diffuse_width_, diffuse_height_, 3};
    CudaTexture normal_tex = {d_normal_data_, normal_width_, normal_height_, 3};
    CudaTexture specular_tex = {d_specular_data_, specular_width_, specular_height_, 1};
    
    // Launch kernel: 使用2D grid，每個thread處理一個像素
    int box_width = bbmaxx - bbminx + 1;
    int box_height = bbmaxy - bbminy + 1;
    
    // 使用16x16的block size（256 threads），適合大多數GPU
    dim3 blockSize(16, 16);
    dim3 gridSize((box_width + blockSize.x - 1) / blockSize.x,
                  (box_height + blockSize.y - 1) / blockSize.y);
    
    render_triangle_kernel_optimized<<<gridSize, blockSize>>>(
        triangle, diffuse_tex, normal_tex, specular_tex, light_dir,
        d_framebuffer_, d_zbuffer_, width_, height_,
        abc_inv_transpose, bbminx, bbmaxx, bbminy, bbmaxy
    );
    
    // 不需要每次都同步，讓GPU流水線工作
    // CUDA_CHECK(cudaGetLastError());
}

void CudaRenderer::download_framebuffer(TGAImage& framebuffer) {
    std::vector<unsigned char> fb_data(width_ * height_ * 3);
    CUDA_CHECK(cudaMemcpy(fb_data.data(), d_framebuffer_, width_ * height_ * 3, cudaMemcpyDeviceToHost));
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            int idx = (x + y * width_) * 3;
            TGAColor c;
            c.bgra[0] = fb_data[idx];
            c.bgra[1] = fb_data[idx + 1];
            c.bgra[2] = fb_data[idx + 2];
            c.bgra[3] = 255;
            c.bytespp = 4;
            framebuffer.set(x, y, c);
        }
    }
    
    // 最後一次同步確保所有工作完成
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// CudaStreamRenderer implementation - 使用多個streams並行處理
// ============================================================================

CudaStreamRenderer::CudaStreamRenderer(int width, int height, int num_streams) 
    : width_(width), height_(height), num_streams_(num_streams), textures_uploaded_(false),
      d_framebuffer_(nullptr), d_zbuffer_(nullptr),
      d_diffuse_data_(nullptr), d_normal_data_(nullptr), d_specular_data_(nullptr) {
    
    // Allocate GPU memory for framebuffer and zbuffer
    size_t fb_size = width * height * 3;
    size_t zbuf_size = width * height * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_framebuffer_, fb_size));
    CUDA_CHECK(cudaMalloc(&d_zbuffer_, zbuf_size));
    
    // Initialize zbuffer to -1000
    std::vector<double> init_zbuf(width * height, -1000.0);
    CUDA_CHECK(cudaMemcpy(d_zbuffer_, init_zbuf.data(), zbuf_size, cudaMemcpyHostToDevice));
    
    // Create CUDA streams
    streams_ = new cudaStream_t[num_streams_];
    for (int i = 0; i < num_streams_; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
}

CudaStreamRenderer::~CudaStreamRenderer() {
    if (d_framebuffer_) CUDA_CHECK(cudaFree(d_framebuffer_));
    if (d_zbuffer_) CUDA_CHECK(cudaFree(d_zbuffer_));
    if (d_diffuse_data_) CUDA_CHECK(cudaFree(d_diffuse_data_));
    if (d_normal_data_) CUDA_CHECK(cudaFree(d_normal_data_));
    if (d_specular_data_) CUDA_CHECK(cudaFree(d_specular_data_));
    
    // Destroy streams
    for (int i = 0; i < num_streams_; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }
    delete[] streams_;
}

void CudaStreamRenderer::upload_textures(const TGAImage& diffuse, const TGAImage& normal, const TGAImage& specular) {
    // Free old textures if they exist
    if (d_diffuse_data_) CUDA_CHECK(cudaFree(d_diffuse_data_));
    if (d_normal_data_) CUDA_CHECK(cudaFree(d_normal_data_));
    if (d_specular_data_) CUDA_CHECK(cudaFree(d_specular_data_));
    
    diffuse_width_ = diffuse.width();
    diffuse_height_ = diffuse.height();
    normal_width_ = normal.width();
    normal_height_ = normal.height();
    specular_width_ = specular.width();
    specular_height_ = specular.height();
    
    // Prepare texture data on host
    std::vector<unsigned char> diffuse_data(diffuse_width_ * diffuse_height_ * 3);
    std::vector<unsigned char> normal_data(normal_width_ * normal_height_ * 3);
    std::vector<unsigned char> specular_data(specular_width_ * specular_height_ * 1);
    
    for (int y = 0; y < diffuse_height_; y++) {
        for (int x = 0; x < diffuse_width_; x++) {
            TGAColor c = diffuse.get(x, y);
            int idx = (x + y * diffuse_width_) * 3;
            diffuse_data[idx] = c.bgra[0];
            diffuse_data[idx + 1] = c.bgra[1];
            diffuse_data[idx + 2] = c.bgra[2];
        }
    }
    
    for (int y = 0; y < normal_height_; y++) {
        for (int x = 0; x < normal_width_; x++) {
            TGAColor c = normal.get(x, y);
            int idx = (x + y * normal_width_) * 3;
            normal_data[idx] = c.bgra[0];
            normal_data[idx + 1] = c.bgra[1];
            normal_data[idx + 2] = c.bgra[2];
        }
    }
    
    for (int y = 0; y < specular_height_; y++) {
        for (int x = 0; x < specular_width_; x++) {
            TGAColor c = specular.get(x, y);
            int idx = x + y * specular_width_;
            specular_data[idx] = c.bgra[0];
        }
    }
    
    // Upload to GPU
    size_t diffuse_size = diffuse_width_ * diffuse_height_ * 3;
    size_t normal_size = normal_width_ * normal_height_ * 3;
    size_t specular_size = specular_width_ * specular_height_ * 1;
    
    CUDA_CHECK(cudaMalloc(&d_diffuse_data_, diffuse_size));
    CUDA_CHECK(cudaMalloc(&d_normal_data_, normal_size));
    CUDA_CHECK(cudaMalloc(&d_specular_data_, specular_size));
    
    CUDA_CHECK(cudaMemcpy(d_diffuse_data_, diffuse_data.data(), diffuse_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normal_data_, normal_data.data(), normal_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_specular_data_, specular_data.data(), specular_size, cudaMemcpyHostToDevice));
    
    textures_uploaded_ = true;
    
    // Timing is now handled in main.cpp
}

void CudaStreamRenderer::init_framebuffer(const TGAColor& bg_color) {
    std::vector<unsigned char> fb_data(width_ * height_ * 3);
    for (int i = 0; i < width_ * height_; i++) {
        fb_data[i * 3 + 0] = bg_color.bgra[0];
        fb_data[i * 3 + 1] = bg_color.bgra[1];
        fb_data[i * 3 + 2] = bg_color.bgra[2];
    }
    CUDA_CHECK(cudaMemcpy(d_framebuffer_, fb_data.data(), width_ * height_ * 3, cudaMemcpyHostToDevice));
    
    // Timing is now handled in main.cpp
}

void CudaStreamRenderer::render_triangles_batch(
    const std::vector<CudaTriangle>& triangles,
    const vec4& light_dir,
    const mat<4,4>& viewport
) {
    if (!textures_uploaded_) {
        fprintf(stderr, "Error: Textures not uploaded!\n");
        return;
    }
    
    // Create texture structs
    CudaTexture diffuse_tex = {d_diffuse_data_, diffuse_width_, diffuse_height_, 3};
    CudaTexture normal_tex = {d_normal_data_, normal_width_, normal_height_, 3};
    CudaTexture specular_tex = {d_specular_data_, specular_width_, specular_height_, 1};
    
    // 使用streams並行處理多個三角形
    for (size_t i = 0; i < triangles.size(); i++) {
        const CudaTriangle& triangle = triangles[i];
        
        // Compute bounding box
        int bbminx = std::max<int>(std::min({triangle.screen[0].x, triangle.screen[1].x, triangle.screen[2].x}), 0);
        int bbmaxx = std::min<int>(std::max({triangle.screen[0].x, triangle.screen[1].x, triangle.screen[2].x}), width_ - 1);
        int bbminy = std::max<int>(std::min({triangle.screen[0].y, triangle.screen[1].y, triangle.screen[2].y}), 0);
        int bbmaxy = std::min<int>(std::max({triangle.screen[0].y, triangle.screen[1].y, triangle.screen[2].y}), height_ - 1);
        
        if (bbminx > bbmaxx || bbminy > bbmaxy) continue;
        
        // Compute ABC matrix
        mat<3,3> ABC = {{
            {triangle.screen[0].x, triangle.screen[0].y, 1.},
            {triangle.screen[1].x, triangle.screen[1].y, 1.},
            {triangle.screen[2].x, triangle.screen[2].y, 1.}
        }};
        
        double abc_det = ABC.det();
        if (abc_det < 1) continue;
        
        mat<3,3> abc_inv_transpose = ABC.invert_transpose();
        
        // 選擇stream（輪流使用）
        int stream_id = i % num_streams_;
        
        // 計算2D grid尺寸
        int box_width = bbmaxx - bbminx + 1;
        int box_height = bbmaxy - bbminy + 1;
        
        // 根據三角形大小動態調整block size
        dim3 blockSize;
        if (box_width * box_height < 256) {
            // 小三角形：使用8x8 block
            blockSize = dim3(8, 8);
        } else if (box_width * box_height < 4096) {
            // 中等三角形：使用16x16 block
            blockSize = dim3(16, 16);
        } else {
            // 大三角形：使用32x8 block（針對寬屏優化）
            blockSize = dim3(32, 8);
        }
        
        dim3 gridSize((box_width + blockSize.x - 1) / blockSize.x,
                      (box_height + blockSize.y - 1) / blockSize.y);
        
        // 在特定stream上啟動kernel（使用優化的2D版本）
        render_triangle_kernel_optimized<<<gridSize, blockSize, 0, streams_[stream_id]>>>(
            triangle, diffuse_tex, normal_tex, specular_tex, light_dir,
            d_framebuffer_, d_zbuffer_, width_, height_,
            abc_inv_transpose, bbminx, bbmaxx, bbminy, bbmaxy
        );
    }
    
    // 等待所有GPU計算完成
    for (int i = 0; i < num_streams_; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }
    
    // Timing is now handled in main.cpp
}

void CudaStreamRenderer::download_framebuffer(TGAImage& framebuffer) {
    std::vector<unsigned char> fb_data(width_ * height_ * 3);
    CUDA_CHECK(cudaMemcpy(fb_data.data(), d_framebuffer_, width_ * height_ * 3, cudaMemcpyDeviceToHost));
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            int idx = (x + y * width_) * 3;
            TGAColor c;
            c.bgra[0] = fb_data[idx];
            c.bgra[1] = fb_data[idx + 1];
            c.bgra[2] = fb_data[idx + 2];
            c.bgra[3] = 255;
            c.bytespp = 4;
            framebuffer.set(x, y, c);
        }
    }
    
    // Timing is now handled in main.cpp
}

// ===== Tile-based CUDA Renderer Implementation =====

// Optimized tile-based rendering kernel with proper tile culling
__global__ void render_tile_kernel(
    const CudaTriangle* triangles,
    int num_triangles,
    const CudaTexture diffuse_tex,
    const CudaTexture normal_tex,
    const CudaTexture specular_tex,
    const vec4 light_dir,
    unsigned char* framebuffer,
    double* zbuffer,
    int width,
    int height,
    int tile_size
) {
    // 計算全局pixel座標
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 邊界檢查
    if (pixel_x >= width || pixel_y >= height) return;
    
    int pixel_idx = pixel_x + pixel_y * width;
    
    // 計算當前pixel所屬的tile
    int tile_x = pixel_x / tile_size;
    int tile_y = pixel_y / tile_size;
    int tile_min_x = tile_x * tile_size;
    int tile_max_x = min(tile_min_x + tile_size - 1, width - 1);
    int tile_min_y = tile_y * tile_size;
    int tile_max_y = min(tile_min_y + tile_size - 1, height - 1);
    
    // 對於每個pixel，只測試與當前tile相交的三角形
    for (int tri_idx = 0; tri_idx < num_triangles; tri_idx++) {
        const CudaTriangle& triangle = triangles[tri_idx];
        
        // Tile culling: 跳過不與當前tile相交的三角形
        if (triangle.bbmax_x < tile_min_x || triangle.bbmin_x > tile_max_x ||
            triangle.bbmax_y < tile_min_y || triangle.bbmin_y > tile_max_y) {
            continue;
        }
        
        // 使用預計算的逆轉置矩陣計算重心座標
        vec3 bc_screen;
        bc_screen.x = triangle.abc_inv_transpose[0][0] * pixel_x + triangle.abc_inv_transpose[0][1] * pixel_y + triangle.abc_inv_transpose[0][2];
        bc_screen.y = triangle.abc_inv_transpose[1][0] * pixel_x + triangle.abc_inv_transpose[1][1] * pixel_y + triangle.abc_inv_transpose[1][2];
        bc_screen.z = triangle.abc_inv_transpose[2][0] * pixel_x + triangle.abc_inv_transpose[2][1] * pixel_y + triangle.abc_inv_transpose[2][2];
        
        // 檢查是否在三角形內
        if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
        
        // 透視校正重心座標
        vec3 bc_clip;
        bc_clip.x = bc_screen.x / triangle.clip[0].w;
        bc_clip.y = bc_screen.y / triangle.clip[1].w;
        bc_clip.z = bc_screen.z / triangle.clip[2].w;
        double bc_sum = bc_clip.x + bc_clip.y + bc_clip.z;
        bc_clip.x /= bc_sum;
        bc_clip.y /= bc_sum;
        bc_clip.z /= bc_sum;
        
        // 計算深度（NDC z）
        vec4 ndc[3];
        for (int i = 0; i < 3; i++) {
            ndc[i].x = triangle.clip[i].x / triangle.clip[i].w;
            ndc[i].y = triangle.clip[i].y / triangle.clip[i].w;
            ndc[i].z = triangle.clip[i].z / triangle.clip[i].w;
            ndc[i].w = 1.0;
        }
        
        double z = bc_screen.x * ndc[0].z + bc_screen.y * ndc[1].z + bc_screen.z * ndc[2].z;
        
        // Z-buffer test (same logic as CPU version)
        if (z <= zbuffer[pixel_idx]) continue;
        
        zbuffer[pixel_idx] = z;
        
        // Fragment shader計算
        vec2 uv;
        uv.x = triangle.varying_uv[0].x * bc_clip.x + 
               triangle.varying_uv[1].x * bc_clip.y + 
               triangle.varying_uv[2].x * bc_clip.z;
        uv.y = triangle.varying_uv[0].y * bc_clip.x + 
               triangle.varying_uv[1].y * bc_clip.y + 
               triangle.varying_uv[2].y * bc_clip.z;
        
        vec4 n;
        n.x = triangle.varying_nrm[0].x * bc_clip.x + 
              triangle.varying_nrm[1].x * bc_clip.y + 
              triangle.varying_nrm[2].x * bc_clip.z;
        n.y = triangle.varying_nrm[0].y * bc_clip.x + 
              triangle.varying_nrm[1].y * bc_clip.y + 
              triangle.varying_nrm[2].y * bc_clip.z;
        n.z = triangle.varying_nrm[0].z * bc_clip.x + 
              triangle.varying_nrm[1].z * bc_clip.y + 
              triangle.varying_nrm[2].z * bc_clip.z;
        n.w = triangle.varying_nrm[0].w * bc_clip.x + 
              triangle.varying_nrm[1].w * bc_clip.y + 
              triangle.varying_nrm[2].w * bc_clip.z;
        
        n = normalized_device(n);
        
        // 光照計算
        double nl = fmax(0.0, dot_device(n, light_dir));
        vec4 r;
        r.x = n.x * (nl * 2.0) - light_dir.x;
        r.y = n.y * (nl * 2.0) - light_dir.y;
        r.z = n.z * (nl * 2.0) - light_dir.z;
        r.w = 0;
        
        // 取得紋理顏色
        unsigned char diffuse_sample[3];
        sample2D_device(diffuse_tex.data, diffuse_tex.width, diffuse_tex.height, 3, uv, diffuse_sample);
        
        unsigned char specular_sample[3];
        sample2D_device(specular_tex.data, specular_tex.width, specular_tex.height, 3, uv, specular_sample);
        
        double specular = pow(fmax(0.0, r.z), specular_sample[0] / 1.0);
        
        // 計算最終顏色
        for (int i = 0; i < 3; i++) {
            int color_val = (int)(diffuse_sample[i] * (0.4 + 1.0 * nl + specular));
            framebuffer[pixel_idx * 3 + i] = (unsigned char)min(color_val, 255);
        }
    }
}

// CudaTileRenderer implementation
CudaTileRenderer::CudaTileRenderer(int width, int height, int tile_size)
    : width_(width), height_(height), tile_size_(tile_size), textures_uploaded_(false),
      d_framebuffer_(nullptr), d_zbuffer_(nullptr),
      d_diffuse_data_(nullptr), d_normal_data_(nullptr), d_specular_data_(nullptr) {
    
    // 計算需要多少個tiles
    tiles_x_ = (width + tile_size - 1) / tile_size;
    tiles_y_ = (height + tile_size - 1) / tile_size;
    total_tiles_ = tiles_x_ * tiles_y_;
    
    printf("Tile-based renderer initialized: %dx%d resolution, tile_size=%d, tiles=%dx%d (%d total)\n", 
           width, height, tile_size, tiles_x_, tiles_y_, total_tiles_);
    
    // 分配GPU記憶體
    size_t fb_size = width * height * 3;
    size_t zbuf_size = width * height * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_framebuffer_, fb_size));
    CUDA_CHECK(cudaMalloc(&d_zbuffer_, zbuf_size));
    
    // 初始化framebuffer為黑色
    CUDA_CHECK(cudaMemset(d_framebuffer_, 0, fb_size));
    
    // 初始化zbuffer
    std::vector<double> init_zbuf(width * height, -1000.0);
    CUDA_CHECK(cudaMemcpy(d_zbuffer_, init_zbuf.data(), zbuf_size, cudaMemcpyHostToDevice));
}

CudaTileRenderer::~CudaTileRenderer() {
    if (d_framebuffer_) CUDA_CHECK(cudaFree(d_framebuffer_));
    if (d_zbuffer_) CUDA_CHECK(cudaFree(d_zbuffer_));
    if (d_diffuse_data_) CUDA_CHECK(cudaFree(d_diffuse_data_));
    if (d_normal_data_) CUDA_CHECK(cudaFree(d_normal_data_));
    if (d_specular_data_) CUDA_CHECK(cudaFree(d_specular_data_));
}

void CudaTileRenderer::upload_textures(const TGAImage& diffuse, const TGAImage& normal, const TGAImage& specular) {
    // 上傳diffuse texture
    diffuse_width_ = diffuse.width();
    diffuse_height_ = diffuse.height();
    size_t diffuse_size = diffuse_width_ * diffuse_height_ * 3;
    
    std::vector<unsigned char> diffuse_data(diffuse_size);
    for (int y = 0; y < diffuse_height_; y++) {
        for (int x = 0; x < diffuse_width_; x++) {
            TGAColor c = diffuse.get(x, y);
            int idx = (x + y * diffuse_width_) * 3;
            diffuse_data[idx] = c.bgra[0];
            diffuse_data[idx + 1] = c.bgra[1];
            diffuse_data[idx + 2] = c.bgra[2];
        }
    }
    
    CUDA_CHECK(cudaMalloc(&d_diffuse_data_, diffuse_size));
    CUDA_CHECK(cudaMemcpy(d_diffuse_data_, diffuse_data.data(), diffuse_size, cudaMemcpyHostToDevice));
    
    // 上傳normal texture
    normal_width_ = normal.width();
    normal_height_ = normal.height();
    size_t normal_size = normal_width_ * normal_height_ * 3;
    
    std::vector<unsigned char> normal_data(normal_size);
    for (int y = 0; y < normal_height_; y++) {
        for (int x = 0; x < normal_width_; x++) {
            TGAColor c = normal.get(x, y);
            int idx = (x + y * normal_width_) * 3;
            normal_data[idx] = c.bgra[0];
            normal_data[idx + 1] = c.bgra[1];
            normal_data[idx + 2] = c.bgra[2];
        }
    }
    
    CUDA_CHECK(cudaMalloc(&d_normal_data_, normal_size));
    CUDA_CHECK(cudaMemcpy(d_normal_data_, normal_data.data(), normal_size, cudaMemcpyHostToDevice));
    
    // 上傳specular texture
    specular_width_ = specular.width();
    specular_height_ = specular.height();
    size_t specular_size = specular_width_ * specular_height_ * 3;
    
    std::vector<unsigned char> specular_data(specular_size);
    for (int y = 0; y < specular_height_; y++) {
        for (int x = 0; x < specular_width_; x++) {
            TGAColor c = specular.get(x, y);
            int idx = (x + y * specular_width_) * 3;
            specular_data[idx] = c.bgra[0];
            specular_data[idx + 1] = c.bgra[1];
            specular_data[idx + 2] = c.bgra[2];
        }
    }
    
    CUDA_CHECK(cudaMalloc(&d_specular_data_, specular_size));
    CUDA_CHECK(cudaMemcpy(d_specular_data_, specular_data.data(), specular_size, cudaMemcpyHostToDevice));
    
    textures_uploaded_ = true;
    printf("Tile renderer: Textures uploaded - diffuse(%dx%d), normal(%dx%d), specular(%dx%d)\n",
           diffuse_width_, diffuse_height_, normal_width_, normal_height_, specular_width_, specular_height_);
}

void CudaTileRenderer::init_framebuffer(const TGAColor& bg_color) {
    std::vector<unsigned char> fb_data(width_ * height_ * 3);
    for (int i = 0; i < width_ * height_; i++) {
        fb_data[i * 3 + 0] = bg_color.bgra[0];
        fb_data[i * 3 + 1] = bg_color.bgra[1];
        fb_data[i * 3 + 2] = bg_color.bgra[2];
    }
    CUDA_CHECK(cudaMemcpy(d_framebuffer_, fb_data.data(), width_ * height_ * 3, cudaMemcpyHostToDevice));
    
    // 重置zbuffer
    std::vector<double> init_zbuf(width_ * height_, -1000.0);
    CUDA_CHECK(cudaMemcpy(d_zbuffer_, init_zbuf.data(), width_ * height_ * sizeof(double), cudaMemcpyHostToDevice));
}

void CudaTileRenderer::render_triangles_batch(
    const std::vector<CudaTriangle>& triangles,
    const vec4& light_dir,
    const mat<4,4>& viewport
) {
    if (!textures_uploaded_) {
        fprintf(stderr, "Error: Textures not uploaded!\n");
        return;
    }
    
    // 將三角形數據複製到GPU
    CudaTriangle* d_triangles;
    size_t triangles_size = triangles.size() * sizeof(CudaTriangle);
    CUDA_CHECK(cudaMalloc(&d_triangles, triangles_size));
    CUDA_CHECK(cudaMemcpy(d_triangles, triangles.data(), triangles_size, cudaMemcpyHostToDevice));
    
    // 創建紋理結構
    CudaTexture diffuse_tex = {d_diffuse_data_, diffuse_width_, diffuse_height_, 3};
    CudaTexture normal_tex = {d_normal_data_, normal_width_, normal_height_, 3};
    CudaTexture specular_tex = {d_specular_data_, specular_width_, specular_height_, 3};
    
    // 設置適合的block size來避免resource limit
    int block_size = min(tile_size_, 16);  // 限制最大block size為16x16
    dim3 blockSize(block_size, block_size);
    dim3 gridSize((width_ + blockSize.x - 1) / blockSize.x, (height_ + blockSize.y - 1) / blockSize.y);
    
    printf("Launching optimized tile kernel: gridSize=(%d,%d), blockSize=(%d,%d), triangles=%zu\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y, triangles.size());
    printf("Debug: tile_size=%d, tiles=%dx%d, width=%d, height=%d\n", 
           tile_size_, tiles_x_, tiles_y_, width_, height_);
    
    // 清空z-buffer
    std::vector<double> init_zbuf(width_ * height_, -1000.0);
    CUDA_CHECK(cudaMemcpy(d_zbuffer_, init_zbuf.data(), width_ * height_ * sizeof(double), cudaMemcpyHostToDevice));
    
    // 啟動優化的tile-based kernel
    render_tile_kernel<<<gridSize, blockSize>>>(
        d_triangles,
        triangles.size(),
        diffuse_tex,
        normal_tex,
        specular_tex,
        light_dir,
        d_framebuffer_,
        d_zbuffer_,
        width_,
        height_,
        tile_size_
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 檢查kernel執行錯誤
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "Optimized tile kernel launch failed: %s\n", cudaGetErrorString(kernel_err));
    }
    
    CUDA_CHECK(cudaFree(d_triangles));
}

void CudaTileRenderer::download_framebuffer(TGAImage& framebuffer) {
    std::vector<unsigned char> fb_data(width_ * height_ * 3);
    CUDA_CHECK(cudaMemcpy(fb_data.data(), d_framebuffer_, width_ * height_ * 3, cudaMemcpyDeviceToHost));
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            int idx = (x + y * width_) * 3;
            TGAColor c;
            c.bgra[0] = fb_data[idx];
            c.bgra[1] = fb_data[idx + 1];
            c.bgra[2] = fb_data[idx + 2];
            c.bgra[3] = 255;
            c.bytespp = 4;
            framebuffer.set(x, y, c);
        }
    }
}

// Template specializations for different tile sizes
class CudaTileRenderer8 : public CudaTileRenderer {
public:
    CudaTileRenderer8(int width, int height) : CudaTileRenderer(width, height, 8) {}
};

class CudaTileRenderer16 : public CudaTileRenderer {
public:
    CudaTileRenderer16(int width, int height) : CudaTileRenderer(width, height, 16) {}
};

class CudaTileRenderer32 : public CudaTileRenderer {
public:
    CudaTileRenderer32(int width, int height) : CudaTileRenderer(width, height, 32) {}
};

