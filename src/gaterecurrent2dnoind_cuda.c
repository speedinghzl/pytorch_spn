// gaterecurrent2dnoind_cuda.c
#include <THC/THC.h>
#include <math.h>
#include "cuda/gaterecurrent2dnoind_kernel.h"

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

// int my_lib_add_forward_cuda(THCudaTensor *input1, THCudaTensor *input2,
// 		       THCudaTensor *output)
// {
//   if (!THCudaTensor_isSameSizeAs(state, input1, input2))
//     return 0;
//   THCudaTensor_resizeAs(state, output, input1);
//   THCudaTensor_cadd(state, output, input1, 1.0, input2);
//   return 1;
// }

// int my_lib_add_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input)
// {
//   THCudaTensor_resizeAs(state, grad_input, grad_output);
//   THCudaTensor_fill(state, grad_input, 1);
//   return 1;
// }

int GateRecurrent2dnoind_forward_cuda(bool horizontal_, bool reverse_, THCudaTensor * X, THCudaTensor * G1, THCudaTensor * G2, THCudaTensor * G3, THCudaTensor * output)
{
	// Grab the input tensor to flat
	float * X_data = THCudaTensor_data(state, X);
	float * G1_data = THCudaTensor_data(state, G1);
	float * G2_data = THCudaTensor_data(state, G2);
	float * G3_data = THCudaTensor_data(state, G3);
	float * H_data = THCudaTensor_data(state, output);

	// dimensions
	int num_ = THCudaTensor_size(state, X, 0);
	int channels_ = THCudaTensor_size(state, X, 1);
	int height_ = THCudaTensor_size(state, X, 2);
	int width_ = THCudaTensor_size(state, X, 3);

	cudaStream_t stream = THCState_getCurrentStream(state);

	if(horizontal_ && !reverse_) // left to right
	{
		//const int count = height_ * channels_ * num_;
		Forward_left_right(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_);
	}
	else if(horizontal_ && reverse_) // right to left
	{
		Forward_right_left(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_);
	}
	else if(!horizontal_ && !reverse_) // top to bottom
	{
		Forward_top_bottom(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_);
	}
	else
	{
		Forward_bottom_top(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, horizontal_, reverse_);
	}

	return 1;
}

int GateRecurrent2dnoind_backward_cuda(bool horizontal_, bool reverse_, THCudaTensor* top, THCudaTensor* top_grad, THCudaTensor * X, THCudaTensor * G1, THCudaTensor * G2, THCudaTensor * G3, THCudaTensor * X_diff, THCudaTensor * G1_diff, THCudaTensor * G2_diff, THCudaTensor * G3_diff)
{
	//Grab the input tensor to flat
	float * X_data = THCudaTensor_data(state, X);
	float * G1_data = THCudaTensor_data(state, G1);
	float * G2_data = THCudaTensor_data(state, G2);
	float * G3_data = THCudaTensor_data(state, G3);
	float * H_data = THCudaTensor_data(state, top);

	float * H_diff = THCudaTensor_data(state, top_grad);

	float * X_diff = THCudaTensor_data(state, X_diff);
	float * G1_diff = THCudaTensor_data(state, G1_diff);
	float * G2_diff = THCudaTensor_data(state, G2_diff);
	float * G3_diff = THCudaTensor_data(state, G3_diff);

	// dimensions
	int num_ = THCudaTensor_size(state, X, 0);
	int channels_ = THCudaTensor_size(state, X, 1);
	int height_ = THCudaTensor_size(state, X, 2);
	int width_ = THCudaTensor_size(state, X, 3);

	cudaStream_t stream = THCState_getCurrentStream(state);

	if(horizontal_ && ! reverse_) //left to right
	{
		Backward_left_right(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);
	}
	else if(horizontal_ &&  reverse_) //right to left
	{
		Backward_right_left(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);
	}
	else if(!horizontal_ &&  !reverse_) //top to bottom
	{
		Backward_top_bottom(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);
	}
	else {
		Backward_bottom_top(num_, channels_, height_, width_, X_data, G1_data, G2_data, G3_data, H_data, X_diff, G1_diff, G2_diff, G3_diff, H_diff, horizontal_, reverse_);
	}

	return 1;
}
