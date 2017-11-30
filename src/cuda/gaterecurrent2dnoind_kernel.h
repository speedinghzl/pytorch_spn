#ifndef _GATERECURRENT2DNOIND_KERNEL
#define _GATERECURRENT2DNOIND_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int Forward_left_right(int num_, int channels_, int height_, int width_, const float * X, const float * G1, const float * G2, const float * G3, float * H, bool horizontal_, bool reverse_, cudaStream_t stream);

int Forward_right_left(int num_, int channels_, int height_, int width_, const float * X, const float * G1, const float * G2, const float * G3, float * H, bool horizontal_, bool reverse_, cudaStream_t stream);

int Forward_top_bottom(int num_, int channels_, int height_, int width_, const float * X, const float * G1, const float * G2, const float * G3, float * H, bool horizontal_, bool reverse_, cudaStream_t stream);

int Forward_bottom_top(int num_, int channels_, int height_, int width_, const float * X, const float * G1, const float * G2, const float * G3, float * H, bool horizontal_, bool reverse_, cudaStream_t stream);

int Backward_left_right(int num_, int channels_, int height_, int width_, const float * X, const float * G1, const float * G2, const float * G3, const float * H, float * X_diff, float * G1_diff, float * G2_diff, float * G3_diff, float * H_diff, bool horizontal_, bool reverse_, cudaStream_t stream);

int Backward_right_left(int num_, int channels_, int height_, int width_, const float * X, const float * G1, const float * G2, const float * G3, const float * H, float * X_diff, float * G1_diff, float * G2_diff, float * G3_diff, float * H_diff, bool horizontal_, bool reverse_, cudaStream_t stream);

int Backward_top_bottom(int num_, int channels_, int height_, int width_, const float * X, const float * G1, const float * G2, const float * G3, const float * H, float * X_diff, float * G1_diff, float * G2_diff, float * G3_diff, float * H_diff, bool horizontal_, bool reverse_, cudaStream_t stream);

int Backward_bottom_top(int num_, int channels_, int height_, int width_, const float * X, const float * G1, const float * G2, const float * G3, const float * H, float * X_diff, float * G1_diff, float * G2_diff, float * G3_diff, float * H_diff, bool horizontal_, bool reverse_, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
