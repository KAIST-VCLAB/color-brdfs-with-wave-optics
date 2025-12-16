#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define FORWARD_NUM_THREADS 512
#define TOTAL_NUM_FREQUENCIES 16       // Best is 16
#define SELECTED_NUM_FREQUENCIES 2      // Best is 2
#define NUM_CHANNELS 3
#define NUM_DIMENSIONS 2
#define BLOCKS_X 16
#define BLOCKS_Y 16
#define PI 3.1415926535626

__device__ float3 operator*(const float a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ void operator+=(float3 &a, const float3 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ float2 operator-(const float2 &a, const float2 &b) {
    return make_float2(a.x-b.x, a.y-b.y);
}

__device__ float2 operator-=(float2 &a, const float2 &b) {
    a.x -= b.x;
    a.y -= b.y;
}

__device__ __forceinline__ bool get256bitOffset(const uint32_t bits[8], const int bitNo){
    int i = bitNo / 32;
    int shift = bitNo % 32;
    return ((bits[i]>>shift)&(uint32_t)1) == 1;    
}

__device__ __forceinline__ void set256bitOffset(uint32_t bits[8], const int bitNo){
    int i = bitNo / 32;
    int shift = bitNo % 32;
    bits[i] |= ((uint32_t)1 << shift);
}


__global__ void point_to_block_and_index(  
    size_t num_points,
    float2 min_position, float2 range,
    const float* __restrict__ positions,
    uint32_t* block,
    uint32_t* point_index
) {
    // Get block/thread related numbers   
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;

    for(auto i = index; i < num_points; i += stride){
        float px = positions[2*i];
        float py = positions[2*i+1];
        int x_block = BLOCKS_X*((px - min_position.x)/range.x);
        int y_block = BLOCKS_Y*((py - min_position.y)/range.y);
        x_block = min(max(x_block, 0), BLOCKS_X-1);
        y_block = min(max(y_block, 0), BLOCKS_Y-1);
        block[i] = y_block*BLOCKS_X+x_block;
        point_index[i] = i;
    }
}

// here scale is in 1D to fit our modeling of Gabor kernel
__global__ void find_blocks_per_gaussian(  
    int num_points,
    float2 min_position, 
    float2 range,
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    int* blocks_per_gaussian,
    float sigma_p
) {
    // Get block/thread related numbers   
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;

    for(int i = index; i < num_points; i += stride){
        // for 1D scale sx==sy
        float s = scales[i];
        float px = positions[2*i];
        float py = positions[2*i+1];
        // for our scale = sigma, r=3sigma=3*s
        // if take sigma_p into consideration
        float r = max(3.0f*s,3.0f*sigma_p);
        int x_block_min = BLOCKS_X*(px - r - min_position.x)/range.x;
        int x_block_max = BLOCKS_X*(px + r - min_position.x)/range.x;
        int y_block_min = BLOCKS_Y*(py - r - min_position.y)/range.y;
        int y_block_max = BLOCKS_Y*(py + r - min_position.y)/range.y;
        
        x_block_min = min(max(0, x_block_min), BLOCKS_X);
        y_block_min = min(max(0, y_block_min), BLOCKS_Y);
        x_block_max = max(min(BLOCKS_X-1, x_block_max), -1);
        y_block_max = max(min(BLOCKS_Y-1, y_block_max), -1);

        blocks_per_gaussian[i] = (x_block_max-x_block_min+1)*(y_block_max-y_block_min+1);
    }
}

// here scale is in 1D to fit our modeling of Gabor kernel
__global__ void create_gaussian_instances(  
    int num_gaussians,
    float2 min_pos, float2 range,
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    uint32_t* __restrict__ cumulative_sums,
    uint32_t* __restrict__ unsorted_gaussian_keys,
    uint32_t* __restrict__ unsorted_gaussian_indices,
    float sigma_p
    ) {
        // Get block/thread related numbers   
        const auto index = blockIdx.x * blockDim.x + threadIdx.x;
        const auto stride = blockDim.x * gridDim.x;

        auto offset = 0;
        for(auto i = index; i < num_gaussians; i += stride){
            offset = (i == 0) ? 0 : cumulative_sums[i-1];
            // for 1D scale sx==sy
            float s = scales[i];
            float px = positions[2*i];
            float py = positions[2*i+1];
            // for our scale = sigma, r=3sigma=3*s
            // if take sigma_p into consideration
            float r = max(3.0f*s,3.0f*sigma_p);
            int x_block_min = BLOCKS_X*(px - r - min_pos.x)/range.x;
            int x_block_max = BLOCKS_X*(px + r - min_pos.x)/range.x;
            int y_block_min = BLOCKS_Y*(py - r - min_pos.y)/range.y;
            int y_block_max = BLOCKS_Y*(py + r - min_pos.y)/range.y;
            
            x_block_min = min(max(0, x_block_min), BLOCKS_X);
            y_block_min = min(max(0, y_block_min), BLOCKS_Y);
            x_block_max = max(min(BLOCKS_X-1, x_block_max), -1);
            y_block_max = max(min(BLOCKS_Y-1, y_block_max), -1);
            for (int x = x_block_min; x <= x_block_max && x < BLOCKS_X; x++){
                for (int y = y_block_min; y <= y_block_max && y < BLOCKS_Y; y++){
                    uint32_t key = (y*BLOCKS_X+x);
                    unsorted_gaussian_keys[offset] = key;
                    unsorted_gaussian_indices[offset] = i;
                    offset++;
                }
            }
        }
    }

__global__ void key_start_end_indices_cuda(size_t num_instances, uint32_t* keys, uint32_t* tile_start_end)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;

    for(auto i = index; i < num_instances; i += stride){
        auto this_key = keys[i];

        if(i > 0){       
            auto last_key = keys[i-1];
            if(this_key != last_key){
                tile_start_end[2*last_key+1] = i;
                tile_start_end[2*this_key] = i;
            }
        }
        if(i < num_instances-1){            
            auto next_key = keys[i+1];
            if(this_key != next_key){
                tile_start_end[2*this_key+1] = i+1;
                tile_start_end[2*next_key] = i+1;
            }
        }
        else{
            tile_start_end[2*this_key+1] = num_instances;
        }
    }
}

__global__ void gabor_render_forward_cuda_kernel(  
    int num_query_points, int num_gaussians, int num_gaussian_instances,   
    const float* __restrict__ query_location,
    const float* __restrict__ query_psi,
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    const float* __restrict__ gabor_a,
    const float* __restrict__ gabor_c,
    const uint32_t* __restrict__ gaussian_instance_indices,
    const uint32_t* __restrict__ block_start_end_index_gaussians,
    const uint32_t* __restrict__ query_indices,
    const uint32_t* __restrict__ block_start_end_index_query_points,
    float* __restrict__ output,
    const float sigma_p
    ) {

    // Get block/thread related numbers   
    const auto threadID = threadIdx.x;
    const auto block_x = blockIdx.x;
    const auto block_y = blockIdx.y;
    const auto this_block_idx = BLOCKS_X*block_y + block_x;

    auto query_point_start_idx = block_start_end_index_query_points[2*this_block_idx];
    auto query_point_end_idx = block_start_end_index_query_points[2*this_block_idx+1];
    auto gaussian_start_idx = block_start_end_index_gaussians[2*this_block_idx];
    auto gaussian_end_idx = block_start_end_index_gaussians[2*this_block_idx+1];

    // return if no query points or gaussians in this block
    if(query_point_start_idx == query_point_end_idx || gaussian_start_idx == gaussian_end_idx) return;

    __shared__ float gaussian_positions[FORWARD_NUM_THREADS][2];
    __shared__ float gaussian_scales[FORWARD_NUM_THREADS];
    __shared__ float gabor_a_shared[FORWARD_NUM_THREADS][2];
    __shared__ float gabor_c_shared[FORWARD_NUM_THREADS][2];
    
    int num_point_batchs = 1 + (gaussian_end_idx - gaussian_start_idx) / FORWARD_NUM_THREADS;

    for(int batch = 0; batch < num_point_batchs; batch++){

        int end_idx_this_batch = min(FORWARD_NUM_THREADS, gaussian_end_idx-gaussian_start_idx-batch*FORWARD_NUM_THREADS);

        // Each thread loads a part of global memory to shared (random reads)
        int collect_idx = gaussian_start_idx + batch*FORWARD_NUM_THREADS + threadID;
        __syncthreads();
        if(collect_idx < num_gaussian_instances){
            int idx = gaussian_instance_indices[collect_idx];
            gaussian_positions[threadID][0] = positions[2*idx];
            gaussian_positions[threadID][1] = positions[2*idx+1];
            gaussian_scales[threadID] = scales[idx];
            gabor_a_shared[threadID][0] = gabor_a[2*idx];
            gabor_a_shared[threadID][1] = gabor_a[2*idx+1];
            gabor_c_shared[threadID][0] = gabor_c[2*idx];
            gabor_c_shared[threadID][1] = gabor_c[2*idx+1];
        }
        __syncthreads();
        // Iterate over all query points this thread is responsible for
        // Update its value according to the currently cached gaussians
        for(auto i = query_point_start_idx+threadID; i < query_point_end_idx; i += FORWARD_NUM_THREADS){
            
            auto real_query_index = query_indices[i];
            float2 x = {query_location[2*real_query_index], query_location[2*real_query_index+1]};
            float2 temp_result = {0.0f, 0.0f};
            for(auto j = 0; j < end_idx_this_batch; j++){
                float2 dx = x - make_float2(gaussian_positions[j][0], gaussian_positions[j][1]);
                float pDistSqr=dx.x*dx.x+dx.y*dx.y;
                // if(pDistSqr>9*sigma_p*sigma_p)
                // {
                //     continue;
                // }
                float coherence_ceoff=(1.0 / (__fsqrt_rn(PI) * sigma_p)) * __expf(-pDistSqr / (2.0 * sigma_p * sigma_p));

                /* linqi yan's method to calculate FT(Gabor)
                // calculate 4 parameters for FT(Gabor)
                // 1.0 / (2.0 * math.pi * self.sigma)
                float sigmaPrime = 1.0f / (2.0f * PI * gaussian_scales[j]);
                // cnis(2.0 * math.pi * (self.a*self.mu).sum(dim=-1))
                // e^(-ix) = cos x - i sin x
                float a_multi_mu=gabor_a_shared[j][0]*gaussian_positions[j][0]+gabor_a_shared[j][1]*gaussian_positions[j][1];
                float2 cnis_a={__cosf(2.0f*PI*a_multi_mu),
                                __sinf(2.0f*PI*a_multi_mu)*(-1.0f)};
                //1.0 / (2.0 * math.pi * self.sigma * self.sigma)) *cnis_a
                cnis_a.x=(float)(1.0f/(2.0f*PI*gaussian_scales[j]*gaussian_scales[j]))*cnis_a.x;
                cnis_a.y=(float)(1.0f/(2.0f*PI*gaussian_scales[j]*gaussian_scales[j]))*cnis_a.y;
                float2 CPrime={gabor_c_shared[j][0]*cnis_a.x-gabor_c_shared[j][1]*cnis_a.y,
                                gabor_c_shared[j][0]*cnis_a.y+gabor_c_shared[j][1]*cnis_a.x};
                float2 muPrime={-1.0f*gabor_a_shared[j][0],
                                -1.0f*gabor_a_shared[j][1]};
                float2 aPrime={gaussian_positions[j][0],
                                gaussian_positions[j][1]};

                
                float adots=aPrime.x*query_psi[2*real_query_index]+aPrime.y*query_psi[2*real_query_index+1];
                // exp_complex=cnis((2.0 * math.pi * aDots)).reshape(N,M,2)
                float2 exp_complex={__cosf(2.0f*PI*adots),
                                __sinf(2.0f*PI*adots)*(-1.0f)};
                float2 complex_term={exp_complex.x*CPrime.x-exp_complex.y*CPrime.y,
                                    exp_complex.x*CPrime.y+exp_complex.y*CPrime.x};
                //1.0 / (math.sqrt(2.0 * math.pi) * sigma) * torch.exp(-0.5 * torch.pow((x - mu) / sigma, 2.0))
                float guassian_term1=1.0f/(__fsqrt_rn(2.0f*PI)*sigmaPrime)*__expf(-0.5f*((query_psi[2*real_query_index] - muPrime.x) / sigmaPrime)*((query_psi[2*real_query_index] - muPrime.x) / sigmaPrime));
                float guassian_term2=1.0f/(__fsqrt_rn(2.0f*PI)*sigmaPrime)*__expf(-0.5f*((query_psi[2*real_query_index+1] - muPrime.y) / sigmaPrime)*((query_psi[2*real_query_index+1] - muPrime.y) / sigmaPrime));

                temp_result.x += complex_term.x*guassian_term1*guassian_term2*coherence_ceoff;
                temp_result.y += complex_term.y*guassian_term1*guassian_term2*coherence_ceoff;
                */
                // do xform calculation in another way which is faster
                float2 v_plus_a={gabor_a_shared[j][0]+query_psi[2*real_query_index],gabor_a_shared[j][1]+query_psi[2*real_query_index+1]};
                float v_plus_a_square=v_plus_a.x*v_plus_a.x+v_plus_a.y*v_plus_a.y;
                float gaussian_term=__expf(-2.0f*PI*PI*gaussian_scales[j]*gaussian_scales[j]*v_plus_a_square);
                float u_mut_va=v_plus_a.x*gaussian_positions[j][0]+v_plus_a.y*gaussian_positions[j][1];
                float2 complex_term={__cosf(u_mut_va*2.0f*PI),__sinf(u_mut_va*2.0f*PI)*(-1.0f)};
                float2 A={complex_term.x*gaussian_term,complex_term.y*gaussian_term};
                float2 A_mul_C={A.x*gabor_c_shared[j][0]-A.y*gabor_c_shared[j][1],
                                A.y*gabor_c_shared[j][0]+A.x*gabor_c_shared[j][1]};
                temp_result.x += A_mul_C.x*coherence_ceoff;
                temp_result.y += A_mul_C.y*coherence_ceoff;
            }
            output[0*num_query_points+real_query_index] += temp_result.x;
            output[1*num_query_points+real_query_index] += temp_result.y;
        }      
    }
}

__global__ void gabor_render_backward_cuda_kernel(
    const int num_primitives,
    const int batch_size, 
    const float* __restrict__ grad_output,
    const float* __restrict__ query_location,
    const float* __restrict__ query_psi,
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    const float* __restrict__ gabor_a,
    const float* __restrict__ gabor_c,
    const float sigma_p,
    float* __restrict__ dPositions,
    float* __restrict__ dScales,
    float* __restrict__ dGabor_a,
    float* __restrict__ dGabor_c,
    const uint32_t* __restrict__ gaussian_instance_indices,
    const uint32_t* __restrict__ block_start_end_index_gaussians,
    const uint32_t* __restrict__ query_indices,
    const uint32_t* __restrict__ block_start_end_index_query_points
    ) 
{
   // Get block/thread related numbers   
    const auto threadID = threadIdx.x;
    const auto block_x = blockIdx.x;
    const auto block_y = blockIdx.y;
    const auto this_block_idx = BLOCKS_X*block_y + block_x;

    auto query_point_start_idx = block_start_end_index_query_points[2*this_block_idx];
    auto query_point_end_idx = block_start_end_index_query_points[2*this_block_idx+1];
    auto gaussian_start_idx = block_start_end_index_gaussians[2*this_block_idx];
    auto gaussian_end_idx = block_start_end_index_gaussians[2*this_block_idx+1];

    // return if no query points or gaussians in this block
    if(query_point_start_idx == query_point_end_idx || gaussian_start_idx == gaussian_end_idx) return;

    __shared__ float query_point_positions[FORWARD_NUM_THREADS][2];
    __shared__ float query_point_psi[FORWARD_NUM_THREADS][2];
    __shared__ float grad_BRDF[FORWARD_NUM_THREADS][2];
    
    auto num_point_batchs = 1 + (query_point_end_idx - query_point_start_idx) / FORWARD_NUM_THREADS;

    for(auto batch = 0; batch < num_point_batchs; batch++){

        auto end_idx_this_batch = min(FORWARD_NUM_THREADS, query_point_end_idx-query_point_start_idx-batch*FORWARD_NUM_THREADS);

        // Each thread loads a part of global memory to shared (random reads)
        auto collect_idx = query_point_start_idx + batch*FORWARD_NUM_THREADS + threadID;
        __syncthreads();
        if(collect_idx < batch_size){
            auto idx = query_indices[collect_idx];
            query_point_positions[threadID][0] = query_location[2*idx];
            query_point_positions[threadID][1] = query_location[2*idx+1];
            query_point_psi[threadID][0] = query_psi[2*idx];
            query_point_psi[threadID][1] = query_psi[2*idx+1];
            grad_BRDF[threadID][0] = grad_output[2*idx];
            grad_BRDF[threadID][1] = grad_output[2*idx+1];
        }
        __syncthreads();
        // Iterate over all gaussians points this thread is responsible for
        // Update its value according to the currently cached gaussians
        for(auto i = gaussian_start_idx+threadID; i < gaussian_end_idx; i += FORWARD_NUM_THREADS){
            
            auto real_query_index = gaussian_instance_indices[i];
            float2 pos = {positions[2*real_query_index], positions[2*real_query_index+1]};
            float s = scales[real_query_index];

            float2 dPosition_temp = {0.0f, 0.0f};
            float dScale_temp = {0.0f};
            float2 da_temp = {0.0f, 0.0f};
            float2 dc_temp = {0.0f, 0.0f};

            for(int j = 0; j < end_idx_this_batch; j++){
                float2 x = {query_point_positions[j][0], query_point_positions[j][1]};
                float2 dx = x - pos;
                float pDistSqr=dx.x*dx.x+dx.y*dx.y;
                // if(pDistSqr>9*sigma_p*sigma_p)
                // {
                //     continue;
                // }
                // calculate the grad from coherence_ceoff
                float coherence_ceoff=(1.0f / (__fsqrt_rn(PI) * sigma_p)) * __expf(-pDistSqr / (2.0f * sigma_p * sigma_p));
                float2 dL_dBRDF={grad_BRDF[j][0]*coherence_ceoff,grad_BRDF[j][1]*coherence_ceoff};

                // do xform calculation in another way
                float2 v_plus_a={gabor_a[2*real_query_index]+query_point_psi[j][0],gabor_a[2*real_query_index+1]+query_point_psi[j][1]};
                float v_plus_a_square=v_plus_a.x*v_plus_a.x+v_plus_a.y*v_plus_a.y;
                float gaussian_term=__expf(-2.0f*PI*PI*s*s*v_plus_a_square);
                float u_mut_va=v_plus_a.x*pos.x+v_plus_a.y*pos.y;
                float2 complex_term={__cosf(u_mut_va*2.0f*PI),__sinf(u_mut_va*2.0f*PI)*(-1.0f)};
                float2 A={complex_term.x*gaussian_term,complex_term.y*gaussian_term};
                float2 A_mul_C={A.x*gabor_c[2*real_query_index]-A.y*gabor_c[2*real_query_index+1],
                                A.y*gabor_c[2*real_query_index]+A.x*gabor_c[2*real_query_index+1]};

                // calculate the grad of coeff branch
                float dL_dcoeff=grad_BRDF[j][0]*A_mul_C.x+grad_BRDF[j][1]*A_mul_C.y;
                dPosition_temp.x+=dL_dcoeff*coherence_ceoff*(x.x-pos.x)/(sigma_p * sigma_p);
                dPosition_temp.y+=dL_dcoeff*coherence_ceoff*(x.y-pos.y)/(sigma_p * sigma_p);

                // calculate the grad from complex number C
                dc_temp.x+=dL_dBRDF.x*A.x+dL_dBRDF.y*A.y;
                dc_temp.y+=-dL_dBRDF.x*A.y+dL_dBRDF.y*A.x;
                float2 dL_dA={dL_dBRDF.x*gabor_c[2*real_query_index]+dL_dBRDF.y*gabor_c[2*real_query_index+1],
                            -dL_dBRDF.x*gabor_c[2*real_query_index+1]+dL_dBRDF.y*gabor_c[2*real_query_index]};
                
                // calculate the grad of the complex term branch
                float2 dL_dcomplex_term={dL_dA.x*gaussian_term,dL_dA.y*gaussian_term};
                float dL_du_mut_va=dL_dcomplex_term.x*2.0f*PI*complex_term.y-dL_dcomplex_term.y*2.0f*PI*complex_term.x;
                dPosition_temp.x+=dL_du_mut_va*v_plus_a.x;
                dPosition_temp.y+=dL_du_mut_va*v_plus_a.y;
                da_temp.x+=dL_du_mut_va*pos.x;
                da_temp.y+=dL_du_mut_va*pos.y;
                
                // caculate the grad of the gaussian term branch
                float dL_dgaussian_term=dL_dA.x*complex_term.x+dL_dA.y*complex_term.y;
                dScale_temp+=dL_dgaussian_term*gaussian_term*2.0f*s*(-2.0f*PI*PI*v_plus_a_square);
                da_temp.x+=dL_dgaussian_term*gaussian_term*2.0f*(gabor_a[2*real_query_index]+query_point_psi[j][0])*(-2.0f*PI*PI*s*s);
                da_temp.y+=dL_dgaussian_term*gaussian_term*2.0f*(gabor_a[2*real_query_index+1]+query_point_psi[j][1])*(-2.0f*PI*PI*s*s);
            }
            atomicAdd(&dPositions[2*real_query_index], dPosition_temp.x);
            atomicAdd(&dPositions[2*real_query_index+1], dPosition_temp.y);
            atomicAdd(&dScales[real_query_index], dScale_temp);
            atomicAdd(&dGabor_a[2*real_query_index], da_temp.x);
            atomicAdd(&dGabor_a[2*real_query_index+1], da_temp.y);
            atomicAdd(&dGabor_c[2*real_query_index], dc_temp.x);
            atomicAdd(&dGabor_c[2*real_query_index+1], dc_temp.y);
        }      
    }
}

void sort_query_points_to_blocks(torch::Tensor positions,
    float2 min_pos, float2 max_pos, 
    uint32_t *&sorted_point_indices, uint32_t *&query_point_block_start_end_indices){
    auto num_points = positions.size(0);
    float2 range = make_float2(max_pos.x-min_pos.x,
                                max_pos.y-min_pos.y);
    
    
    uint32_t *unsorted_point_blocks, *unsorted_point_indices;
    cudaMalloc((void**)&unsorted_point_blocks, num_points*sizeof(uint32_t));  
    cudaMalloc((void**)&unsorted_point_indices, num_points*sizeof(uint32_t));
    point_to_block_and_index<<<num_points+512-1/512, 512>>>(num_points,
        min_pos, range,
        positions.contiguous().data_ptr<float>(),
        unsorted_point_blocks, unsorted_point_indices);

    uint32_t *sorted_point_blocks;
    cudaMalloc((void**)&sorted_point_blocks, num_points*sizeof(uint32_t));  
    cudaMalloc((void**)&sorted_point_indices, num_points*sizeof(uint32_t));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_point_blocks, sorted_point_blocks,
		unsorted_point_indices, sorted_point_indices,
		num_points);
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_point_blocks, sorted_point_blocks,
		unsorted_point_indices, sorted_point_indices,
		num_points);


    cudaMalloc((void**)&query_point_block_start_end_indices, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(int));   
    cudaMemset(query_point_block_start_end_indices, 0, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(int));
    key_start_end_indices_cuda<<<(num_points + 512 - 1) / 512, 512>>> (
        num_points,
        sorted_point_blocks,
        query_point_block_start_end_indices
        );    
    cudaFree(unsorted_point_blocks);
    cudaFree(unsorted_point_indices);
    cudaFree(sorted_point_blocks);
    cudaFree(d_temp_storage);
}

// here gaussian_scales is 1D to fit our modeling of Gabor Kernel
uint32_t sort_gaussians_to_blocks(torch::Tensor gaussian_positions, torch::Tensor gaussian_scales,
    float2 min_pos, float2 max_pos, 
    uint32_t *&sorted_gaussian_indices, uint32_t *&block_start_end_indices, float sigma_p){

    // 1. Determine the number of gaussians per block
    auto num_gaussians = gaussian_positions.size(0);
    float2 range = make_float2(max_pos.x-min_pos.x,max_pos.y-min_pos.y);
    int* blocks_per_gaussian;
    cudaMalloc((void**)&blocks_per_gaussian, num_gaussians*sizeof(int));   

    find_blocks_per_gaussian<<<(num_gaussians+512-1)/512,512>>>(
        num_gaussians, min_pos, range,
        gaussian_positions.contiguous().data_ptr<float>(),
        gaussian_scales.contiguous().data_ptr<float>(),
        blocks_per_gaussian,
        sigma_p
        );

    // 2. Inclusive sum on gaussians per block to find total number
    // of gaussian instances needed
    // Allocate temp storage for the inclusive sum
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    uint32_t* cumulative_sums;
    cudaMalloc((void**)&cumulative_sums, num_gaussians*sizeof(uint32_t));    
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_gaussian, cumulative_sums, num_gaussians);    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_gaussian, cumulative_sums, num_gaussians);  

    
    
    // Get the total number of gaussian instances we have on host (cpu)
    uint32_t total_gaussian_instances;
    cudaMemcpy(&total_gaussian_instances, &cumulative_sums[num_gaussians-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // If 0 gaussians need to be rendered, return
    if(total_gaussian_instances == 0) return total_gaussian_instances;
    // 3. Create the gaussian instances
    uint32_t *unsorted_gaussian_keys, *sorted_gaussian_keys;
    uint32_t *unsorted_gaussian_indices;
    cudaMalloc((void**)&unsorted_gaussian_keys, total_gaussian_instances*sizeof(uint32_t));   
    cudaMalloc((void**)&sorted_gaussian_keys, total_gaussian_instances*sizeof(uint32_t));   
    cudaMalloc((void**)&unsorted_gaussian_indices, total_gaussian_instances*sizeof(uint32_t));  
    cudaMalloc((void**)&sorted_gaussian_indices, total_gaussian_instances*sizeof(uint32_t));   

    create_gaussian_instances<<<(num_gaussians+512-1)/512,512>>>(
        num_gaussians,
        min_pos, range, 
        gaussian_positions.contiguous().data_ptr<float>(),
        gaussian_scales.contiguous().data_ptr<float>(),
        cumulative_sums,
        unsorted_gaussian_keys,
        unsorted_gaussian_indices,
        sigma_p
    );

    // 4. Sort the gaussian instances by keys (tileID)
    cudaFree(d_temp_storage);
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_gaussian_keys, sorted_gaussian_keys,
		unsorted_gaussian_indices, sorted_gaussian_indices,
		total_gaussian_instances);

    // Then actually sort
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_gaussian_keys, sorted_gaussian_keys,
		unsorted_gaussian_indices, sorted_gaussian_indices,
		total_gaussian_instances);

    // 5. Identify index start/end index for gaussians in each tile 
    cudaMalloc((void**)&block_start_end_indices, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(int));   
    cudaMemset(block_start_end_indices, 0, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(int));
    key_start_end_indices_cuda<<<(total_gaussian_instances + 512 - 1) / 512, 512>>> (
        total_gaussian_instances,
        sorted_gaussian_keys,
        block_start_end_indices
        );
    // Only relevant memory is block_start_end_indices and sorted_gaussian_indices.
    // Free the rest.
    cudaFree(blocks_per_gaussian);
    cudaFree(d_temp_storage);
    cudaFree(unsorted_gaussian_indices);
    cudaFree(unsorted_gaussian_keys);
    cudaFree(sorted_gaussian_keys);
    cudaFree(cumulative_sums);
    return total_gaussian_instances;
}

std::vector<torch::Tensor> gabor_render_forward_cuda(
    const torch::Tensor& query_location,        // [N, 2]
    const torch::Tensor& query_psi,             // [N, 2]
    const torch::Tensor& positions,       // [M, 2]
    const torch::Tensor& scales,          // [M, 1]
    const torch::Tensor& gabor_a,               // [M, 2]
    const torch::Tensor& gabor_c,               // [M, 2]
    const float sigma_p,
    const float H,
    const float W
    ) {
    
    auto num_query_points = query_location.size(0);
    auto num_gaussians = positions.size(0);

    // Create output tensor and other tensors
    auto output = torch::zeros({2, query_location.size(0)}, query_location.device());
    
    // Sort query points and gaussians into 16x16 blocks
    uint32_t* sorted_gaussian_indices;
    uint32_t* blocks_gaussian_start_end_indices;
    uint32_t num_gaussian_instances = sort_gaussians_to_blocks(
        positions, scales,
        make_float2(0.0f, 0.0f), make_float2((float)H, (float)W),
        sorted_gaussian_indices, 
        blocks_gaussian_start_end_indices, sigma_p);
    
    // for position, no change is needed for our problem
    uint32_t* sorted_query_point_indices;
    uint32_t* blocks_query_points_start_end_indices;
    sort_query_points_to_blocks(
        query_location, make_float2(0.0f, 0.0f), make_float2((float)H, (float)W),
        sorted_query_point_indices, 
        blocks_query_points_start_end_indices);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor sorted_gaussian_indices_tensor = torch::from_blob(sorted_gaussian_indices, {num_gaussian_instances}, options).clone();
    torch::Tensor blocks_gaussian_start_end_indices_tensor = torch::from_blob(blocks_gaussian_start_end_indices, {BLOCKS_X*BLOCKS_Y, 2}, options).clone();
    torch::Tensor sorted_query_point_indices_tensor = torch::from_blob(sorted_query_point_indices, {query_location.size(0)}, options).clone();
    torch::Tensor blocks_query_points_start_end_indices_tensor = torch::from_blob(blocks_query_points_start_end_indices, {BLOCKS_X*BLOCKS_Y, 2}, options).clone();
    
    if(num_gaussian_instances == 0) return {output, sorted_gaussian_indices_tensor, 
        blocks_gaussian_start_end_indices_tensor, sorted_query_point_indices_tensor, 
        blocks_query_points_start_end_indices_tensor};
    

    // Now sorted_gaussian_indices orders the indices of the original gaussian
    // tensors in block order, so items are in block [0, 0, ..., 0, 1, 1, ..., 1, 2, ...]
    // Similar with sorted_query_point_indices.
    // cumulative_gaussians_per_block and cumulative_query_points_per_block are the
    // indices for which block 0->1 (so each thread block knows where to stop)

    // Finally evaluate results such that query points only evaulate with gaussians
    // within the block.
    dim3 numBlocks (16, 16);
    gabor_render_forward_cuda_kernel<<<numBlocks, FORWARD_NUM_THREADS>>>(
        query_location.size(0), positions.size(0), num_gaussian_instances,
        query_location.contiguous().data_ptr<float>(),
        query_psi.contiguous().data_ptr<float>(),
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        gabor_a.contiguous().data_ptr<float>(),
        gabor_c.contiguous().data_ptr<float>(),
        sorted_gaussian_indices,
        blocks_gaussian_start_end_indices,
        sorted_query_point_indices,
        blocks_query_points_start_end_indices,
        output.contiguous().data_ptr<float>(),
        sigma_p
        );
    cudaFree(sorted_gaussian_indices);
    cudaFree(blocks_gaussian_start_end_indices);
    cudaFree(sorted_query_point_indices);
    cudaFree(blocks_query_points_start_end_indices);
    return {output, sorted_gaussian_indices_tensor, 
        blocks_gaussian_start_end_indices_tensor, 
        sorted_query_point_indices_tensor,
        blocks_query_points_start_end_indices_tensor};
}

std::vector<torch::Tensor> gabor_render_backward_cuda(
    const torch::Tensor& grad_output,           // [N, 2]
    const torch::Tensor& query_location,        // [N, 2]
    const torch::Tensor& query_psi,             // [N, 2]
    const torch::Tensor& positions,       // [M, 2]
    const torch::Tensor& scales,          // [M, 1]
    const torch::Tensor& gabor_a,               // [M, 2]
    const torch::Tensor& gabor_c,               // [M, 2]
    const float sigma_p,
    const torch::Tensor& gabor_instance_indices,
    const torch::Tensor& block_start_end_index_gabors,
    const torch::Tensor& query_indices,
    const torch::Tensor& block_start_end_index_query_points
    ) {
        // Get sizes for the output
        const auto batch_size = query_location.size(0);
        const auto num_primitives = positions.size(0);

        // Set up gradient tensors
        auto dPositions = torch::zeros_like(positions); 
        auto dScales = torch::zeros_like(scales); 
        auto dGabor_a = torch::zeros_like(gabor_a); 
        auto dGabor_c = torch::zeros_like(gabor_c);
        
        //int numSMs;
        //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        //const int blocks = (num_primitives+FORWARD_NUM_THREADS-1)/FORWARD_NUM_THREADS;
        
        dim3 blocks (16, 16);
        gabor_render_backward_cuda_kernel<<<blocks, FORWARD_NUM_THREADS>>>(
            num_primitives,
            batch_size,
            grad_output.contiguous().data_ptr<float>(),
            query_location.contiguous().data_ptr<float>(),
            query_psi.contiguous().data_ptr<float>(),
            positions.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            gabor_a.contiguous().data_ptr<float>(),
            gabor_c.contiguous().data_ptr<float>(),
            sigma_p,
            dPositions.contiguous().data_ptr<float>(),
            dScales.contiguous().data_ptr<float>(),
            dGabor_a.contiguous().data_ptr<float>(),
            dGabor_c.contiguous().data_ptr<float>(),
            (uint32_t*)gabor_instance_indices.contiguous().data_ptr<int>(),
            (uint32_t*)block_start_end_index_gabors.contiguous().data_ptr<int>(),
            (uint32_t*)query_indices.contiguous().data_ptr<int>(),
            (uint32_t*)block_start_end_index_query_points.contiguous().data_ptr<int>()
            );
    
        return {dPositions, dScales, dGabor_a, dGabor_c };
    }