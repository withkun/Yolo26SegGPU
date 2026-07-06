#ifndef __INC_CUDA_SEGMENT_H
#define __INC_CUDA_SEGMENT_H

#include "cuda_runtime.h"


#define PROBES_W            38
#define COEFFS_W            32


void launchNMS(const float *d_proposals, int32_t *d_keep_index, float iou_threshold, float conf_threshold, int32_t M);

void launchGEMV(const float *d_proposals, const float *d_prototypes, const int32_t *d_keep_index, float *d_proto_masks,
                int32_t proto_h, int32_t proto_w, int32_t M, int32_t P, int32_t K, int32_t N);

void launchScale(const float *d_proposals, const int32_t *d_keep_index, const float *d_proto_masks, uint8_t *d_final_masks,
                 int32_t final_steps, int32_t ideal_width, float mask_threshold, int32_t image_h, int32_t image_w,
                 int32_t proto_h, int32_t proto_w, float scale_xy, float sampling, int32_t M, int32_t N);


void cudaPostprocess(const float *d_proposals, const float *d_prototypes, int32_t *d_keep_index, int32_t *h_keep_index, float *d_proto_masks, uint8_t *d_final_masks, uint8_t *h_final_masks,
                     int32_t final_steps, int32_t ideal_width, float iou_threshold, float conf_threshold, float mask_threshold, int32_t image_h, int32_t image_w,
                     int32_t proto_h, int32_t proto_w, float scale_xy, float sampling, int32_t M, int32_t P, int32_t K,
                     int64_t &usage1, int64_t &usage2, int64_t &usage3);

#endif //__INC_CUDA_SEGMENT_H