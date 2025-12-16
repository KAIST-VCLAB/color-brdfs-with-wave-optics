# calculate FT(R) for rendering
import torch
import GaborRender as gabor_render


class GaborRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_location,query_psi,
            gabor_positions, gabor_scales, gabor_a,
            gabor_c,sigma_p,H,W):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        
        outputs = gabor_render.forward(query_location,query_psi,
            gabor_positions, gabor_scales, gabor_a,
            gabor_c,sigma_p,H,W)
        
        result, sorted_gabor_indices_tensor, \
        blocks_gabor_start_end_indices_tensor, \
        sorted_query_point_indices, blocks_query_points_start_end_indices_tensor = outputs
        
        variables = [query_location,query_psi, 
                     gabor_positions, gabor_scales, gabor_a, gabor_c,
                    sorted_gabor_indices_tensor, 
                    blocks_gabor_start_end_indices_tensor, 
                    sorted_query_point_indices, 
                    blocks_query_points_start_end_indices_tensor]
        
        ctx.save_for_backward(*variables)
        ctx.sigma_p=sigma_p
        return result.T

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_output=grad_output.contiguous()
        outputs = gabor_render.backward(grad_output, *ctx.saved_tensors,ctx.sigma_p)
        
        grad_gabor_position, grad_gabor_scales, grad_gabor_a, grad_gabor_c = outputs
        #for t in ctx.saved_tensors:
        #    del t
        return None,None, grad_gabor_position, grad_gabor_scales, \
              grad_gabor_a, grad_gabor_c, None, None, None