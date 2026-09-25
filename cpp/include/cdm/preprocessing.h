/*
* CDM pre- and post-processing.
*
* A C++ copy of what RGBDDepth.infer_depth does around the network
* (camera_depth_models/dpt.py), kept separate from the TensorRT runtime so it
* can be tested without a GPU. Every function names the Python step it
* reproduces; a change on either side must be made on both.
*/

#ifndef CDM_PREPROCESSING_H
#define CDM_PREPROCESSING_H

#include <opencv2/core/core.hpp>

namespace cdm
{

/*! DINOv2 patch size; the network resolution is a multiple of it. */
constexpr int kPatchSize = 14;

/*!
 * Network resolution for a camera image: util.transform.Resize with
 * keep_aspect_ratio, resize_method "lower_bound", ensure_multiple_of 14.
 * 640x480 at input_size 518 gives 686x518.
 */
cv::Size NetworkSize(const cv::Size &image, int input_size);

/*!
 * Fills `dst` (4 x net.height x net.width floats, NCHW without the batch) with
 * the network input infer_depth builds:
 *   channels 0-2  RGB / 255, bicubic-resized, ImageNet mean/std normalized;
 *   channel  3    1 / depth, zero where depth is missing, nearest-resized.
 *
 * bgr     CV_8UC3, as OpenCV reads it.
 * depth_m CV_32FC1 metres, same size; a pixel that is not finite and positive
 *         is missing. Upstream feeds exactly those pixels a zero inverse depth.
 */
void Preprocess(const cv::Mat &bgr, const cv::Mat &depth_m, const cv::Size &net, float *dst);

/*!
 * Metric depth at `image` resolution from the network's inverse depth
 * (net.height x net.width floats): PyTorch "nearest" upsampling, then 1/x.
 *
 * Where the network predicts zero inverse depth, upstream returns +inf; this
 * returns 0, the system's "no depth" value, and counts the pixel in
 * `num_invalid` when given. A non-finite prediction means a broken engine
 * (e.g. an FP16 overflow) and throws rather than reach the map.
 */
cv::Mat Postprocess(const float *inverse_depth, const cv::Size &net, const cv::Size &image,
                       int *num_invalid = nullptr);

/*! Source index PyTorch's nearest interpolation reads for output index `dst`. */
int TorchNearestIndex(int dst, int in_size, int out_size);

} // namespace cdm

#endif
