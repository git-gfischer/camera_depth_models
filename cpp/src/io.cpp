/* CDM apps: depth image I/O and visualisation. */

#include "cdm/io.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace cdm
{

cv::Mat LoadDepthMetres(const std::string &path, double depth_factor)
{
    if(!(depth_factor > 0.0))
        throw std::invalid_argument("depth factor must be positive");
    const cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if(raw.empty())
        throw std::runtime_error("cannot read depth image '" + path + "'");
    if(raw.type() != CV_16UC1)
        throw std::runtime_error("depth image '" + path + "' is not single-channel uint16");
    cv::Mat depth;
    raw.convertTo(depth, CV_32F, 1.0 / depth_factor);
    return depth;
}

cv::Mat DepthToU16(const cv::Mat &depth_m, double depth_factor, int *num_clipped)
{
    if(depth_m.type() != CV_32FC1)
        throw std::invalid_argument("DepthToU16 expects CV_32FC1 metres");
    int clipped = 0;
    cv::Mat out(depth_m.size(), CV_16UC1);
    for(int row = 0; row < depth_m.rows; row++)
    {
        const float *in = depth_m.ptr<float>(row);
        unsigned short *o = out.ptr<unsigned short>(row);
        for(int col = 0; col < depth_m.cols; col++)
        {
            const double raw = std::round(static_cast<double>(in[col]) * depth_factor);
            if(!(raw > 0.0)) o[col] = 0;
            else if(raw > 65535.0) { o[col] = 0; clipped++; }
            else o[col] = static_cast<unsigned short>(raw);
        }
    }
    if(num_clipped != nullptr) *num_clipped = clipped;
    return out;
}

float AutoDepthRange(const cv::Mat &depth_m)
{
    std::vector<float> valid;
    valid.reserve(depth_m.total());
    for(int row = 0; row < depth_m.rows; row++)
    {
        const float *d = depth_m.ptr<float>(row);
        for(int col = 0; col < depth_m.cols; col++)
            if(std::isfinite(d[col]) && d[col] > 0.f) valid.push_back(d[col]);
    }
    if(valid.empty()) return 5.f;
    const size_t k = static_cast<size_t>(0.95 * (valid.size() - 1));
    std::nth_element(valid.begin(), valid.begin() + k, valid.end());
    return 1.1f * valid[k];
}

cv::Mat ColorizeDepth(const cv::Mat &depth_m, float max_m)
{
    cv::Mat scaled, colour;
    depth_m.convertTo(scaled, CV_8U, 255.0 / max_m);
    cv::applyColorMap(scaled, colour, cv::COLORMAP_TURBO);
    colour.setTo(cv::Scalar(0, 0, 0), depth_m <= 0.f);
    return colour;
}

cv::Mat SideBySide(const cv::Mat &bgr, const cv::Mat &raw_m, const cv::Mat &cdm_m, float max_m,
                   const std::string &status)
{
    cv::Mat panels[3] = {bgr.clone(), ColorizeDepth(raw_m, max_m), ColorizeDepth(cdm_m, max_m)};
    const char *captions[3] = {"RGB", "sensor depth", "CDM depth"};
    for(int i = 0; i < 3; i++)
        cv::putText(panels[i], captions[i], cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);
    cv::Mat out;
    cv::hconcat(panels, 3, out);
    if(!status.empty())
        cv::putText(out, status, cv::Point(10, out.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);
    return out;
}

} // namespace cdm
