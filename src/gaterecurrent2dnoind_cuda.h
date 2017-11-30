// gaterecurrent2dnoind_cuda.h
int GateRecurrent2dnoind_forward_cuda(bool horizontal_, bool reverse_, THCudaTensor * X, THCudaTensor * G1, THCudaTensor * G2, THCudaTensor * G3, THCudaTensor * output);

int GateRecurrent2dnoind_backward_cuda(bool horizontal_, bool reverse_, THCudaTensor* top, THCudaTensor* top_grad, THCudaTensor * X, THCudaTensor * G1, THCudaTensor * G2, THCudaTensor * G3, THCudaTensor * X_diff, THCudaTensor * G1_diff, THCudaTensor * G2_diff, THCudaTensor * G3_diff);
