#include <torch/extension.h>
#include <vector>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::vector<torch::Tensor> gabor_render_forward_cuda(
    const torch::Tensor& query_location,        // [N, 2]
    const torch::Tensor& query_psi,             // [N, 2]
    const torch::Tensor& gabor_positions,       // [M, 2]
    const torch::Tensor& gabor_scales,          // [M, 1]
    const torch::Tensor& gabor_a,               // [M, 2]
    const torch::Tensor& gabor_c,               // [M, 2]
    const float sigma_p,
    const float H,
    const float W
);

std::vector<torch::Tensor> gabor_render_backward_cuda(
    const torch::Tensor& grad_output,           // [N, 2]
    const torch::Tensor& query_location,        // [N, 2]
    const torch::Tensor& query_psi,             // [N, 2]
    const torch::Tensor& gabor_positions,       // [M, 2]
    const torch::Tensor& gabor_scales,          // [M, 1]
    const torch::Tensor& gabor_a,               // [M, 2]
    const torch::Tensor& gabor_c,               // [M, 2]
    const float sigma_p,
    const torch::Tensor& gabor_instance_indices,
    const torch::Tensor& block_start_end_index_gabors,
    const torch::Tensor& query_indices,
    const torch::Tensor& block_start_end_index_query_points
    );        


// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> gabor_render_forward(
    const torch::Tensor& query_location,        // [N, 2]
    const torch::Tensor& query_psi,             // [N, 2]
    const torch::Tensor& gabor_positions,       // [M, 2]
    const torch::Tensor& gabor_scales,          // [M, 1]
    const torch::Tensor& gabor_a,               // [M, 2]
    const torch::Tensor& gabor_c,               // [M, 2]
    const float sigma_p,
    const float H,
    const float W
    ) {
    CHECK_INPUT(query_location);
    CHECK_INPUT(query_psi);
    CHECK_INPUT(gabor_positions);
    CHECK_INPUT(gabor_scales);
    CHECK_INPUT(gabor_a);
    CHECK_INPUT(gabor_c);
    /*
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);

    torch::Tensor queryPointBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> queryPointBufferFunct = resizeFunctional(queryPointBuffer);

    torch::Tensor gaussiansBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> gaussiansBufferFunct = resizeFunctional(gaussiansBuffer);
    */
    return gabor_render_forward_cuda(
        query_location, 
        query_psi,
        gabor_positions, 
        gabor_scales,  
        gabor_a,
        gabor_c,
        sigma_p,
        H,
        W
        );
}

std::vector<torch::Tensor> gabor_render_backward(
    const torch::Tensor& grad_output,           // [N, 2]
    const torch::Tensor& query_location,        // [N, 2]
    const torch::Tensor& query_psi,             // [N, 2]
    const torch::Tensor& gabor_positions,       // [M, 2]
    const torch::Tensor& gabor_scales,          // [M, 1]
    const torch::Tensor& gabor_a,               // [M, 2]
    const torch::Tensor& gabor_c,               // [M, 2]
    const torch::Tensor& gabor_instance_indices,
    const torch::Tensor& block_start_end_index_gabors,
    const torch::Tensor& query_indices,
    const torch::Tensor& block_start_end_index_query_points,
    const float sigma_p
    ) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(query_location);
    CHECK_INPUT(query_psi);
    CHECK_INPUT(gabor_positions);
    CHECK_INPUT(gabor_scales);
    CHECK_INPUT(gabor_a);
    CHECK_INPUT(gabor_c);
    CHECK_INPUT(gabor_instance_indices);
    CHECK_INPUT(block_start_end_index_gabors);
    CHECK_INPUT(query_indices);
    CHECK_INPUT(block_start_end_index_query_points);
    
    return gabor_render_backward_cuda(
        grad_output, 
        query_location,
        query_psi,
        gabor_positions, 
        gabor_scales,  
        gabor_a,
        gabor_c,
        sigma_p,
        gabor_instance_indices,
        block_start_end_index_gabors,
        query_indices,
        block_start_end_index_query_points
        );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gabor_render_forward, "gabor_render forward (CUDA)");
  m.def("backward", &gabor_render_backward, "gabor_render backward (CUDA)");
}