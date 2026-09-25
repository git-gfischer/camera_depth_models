/* CDM pre- and post-processing. */

#include "cdm/preprocessing.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace cdm
{

namespace {

/*! NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), RGB order. */
const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

/*!
 * Resize.constrain_to_multiple_of with only min_val set. np.round rounds half
 * to even, which std::nearbyint does under the default rounding mode.
 */
int ConstrainToMultipleOf(double x, int min_val)
{
    const double m = kPatchSize;
    int y = static_cast<int>(std::nearbyint(x / m) * m);
    if(y < min_val) y = static_cast<int>(std::ceil(x / m) * m);
    return y;
}

} // namespace

cv::Size NetworkSize(const cv::Size &image, int input_size)
{
    if(image.width <= 0 || image.height <= 0)
        throw std::invalid_argument("CDM: empty image size");
    if(input_size < kPatchSize)
        throw std::invalid_argument("CDM: input_size must be at least the patch size");

    // Resize.get_size, lower_bound with keep_aspect_ratio: fit the side that
    // needs the larger scale, so both sides end up at least input_size.
    double scale_h = static_cast<double>(input_size) / image.height;
    double scale_w = static_cast<double>(input_size) / image.width;
    if(scale_w > scale_h) scale_h = scale_w;
    else scale_w = scale_h;

    return cv::Size(ConstrainToMultipleOf(scale_w * image.width, input_size),
                    ConstrainToMultipleOf(scale_h * image.height, input_size));
}

void Preprocess(const cv::Mat &bgr, const cv::Mat &depth_m, const cv::Size &net, float *dst)
{
    if(bgr.empty() || bgr.type() != CV_8UC3)
        throw std::invalid_argument("CDM: colour image must be CV_8UC3 (BGR)");
    if(depth_m.type() != CV_32FC1)
        throw std::invalid_argument("CDM: depth must be CV_32FC1 in metres");
    if(depth_m.size() != bgr.size())
        throw std::invalid_argument("CDM: colour and depth images differ in size");
    if(net.width % kPatchSize != 0 || net.height % kPatchSize != 0)
        throw std::invalid_argument("CDM: network size must be a multiple of the patch size");

    const int area = net.area();

    // RGB: /255, bicubic resize, normalize (infer_depth, Resize, NormalizeImage).
    cv::Mat rgb, rgb_f, rgb_net;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb_f, CV_32F, 1.0 / 255.0);
    cv::resize(rgb_f, rgb_net, net, 0, 0, cv::INTER_CUBIC);
    cv::Mat planes[3];
    cv::split(rgb_net, planes);
    for(int c = 0; c < 3; c++)
    {
        cv::Mat plane(net, CV_32FC1, dst + static_cast<size_t>(c) * area);
        planes[c].convertTo(plane, CV_32F, 1.0 / kStd[c], -kMean[c] / kStd[c]);
    }

    // Inverse depth, zero where missing, nearest resize (infer_depth, Resize).
    cv::Mat inverse(depth_m.size(), CV_32FC1);
    for(int row = 0; row < depth_m.rows; row++)
    {
        const float *d = depth_m.ptr<float>(row);
        float *out = inverse.ptr<float>(row);
        for(int col = 0; col < depth_m.cols; col++)
            out[col] = (std::isfinite(d[col]) && d[col] > 0.f) ? 1.f / d[col] : 0.f;
    }
    cv::Mat inverse_net(net, CV_32FC1, dst + static_cast<size_t>(3) * area);
    cv::resize(inverse, inverse_net, net, 0, 0, cv::INTER_NEAREST);
    if(inverse_net.data != reinterpret_cast<uchar*>(dst + static_cast<size_t>(3) * area))
        throw std::logic_error("CDM: resize reallocated the depth plane");
}

int TorchNearestIndex(int dst, int in_size, int out_size)
{
    // upsample_nearest: scale = float(in) / out, src = min(floor(dst * scale), in - 1),
    // in float32. OpenCV's INTER_NEAREST computes the ratio in double and can
    // land one pixel off where dst * scale is an integer, so it is not used.
    const float scale = static_cast<float>(in_size) / static_cast<float>(out_size);
    const int src = static_cast<int>(std::floor(static_cast<float>(dst) * scale));
    return std::min(src, in_size - 1);
}

cv::Mat Postprocess(const float *inverse_depth, const cv::Size &net, const cv::Size &image,
                       int *num_invalid)
{
    if(inverse_depth == nullptr)
        throw std::invalid_argument("CDM: no network output");

    for(size_t i = 0; i < static_cast<size_t>(net.area()); i++)
    {
        if(!std::isfinite(inverse_depth[i]))
            throw std::runtime_error(
                "CDM: non-finite network output; the engine is numerically broken "
                "(rebuild it in FP32)");
    }

    std::vector<int> src_col(image.width);
    for(int col = 0; col < image.width; col++)
        src_col[col] = TorchNearestIndex(col, net.width, image.width);

    int invalid = 0;
    cv::Mat depth(image, CV_32FC1);
    for(int row = 0; row < image.height; row++)
    {
        const float *in = inverse_depth +
            static_cast<size_t>(TorchNearestIndex(row, net.height, image.height)) * net.width;
        float *out = depth.ptr<float>(row);
        for(int col = 0; col < image.width; col++)
        {
            const float inv = in[src_col[col]];
            if(inv > 0.f)
                out[col] = 1.f / inv;
            else
            {
                out[col] = 0.f;
                invalid++;
            }
        }
    }
    if(num_invalid != nullptr) *num_invalid = invalid;
    return depth;
}

} // namespace cdm
