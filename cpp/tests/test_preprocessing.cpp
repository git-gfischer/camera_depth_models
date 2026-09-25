/*
 * The CDM pre/post-processing must be the one infer_depth applies, or the
 * TensorRT engine is fed inputs the network was never trained on.
 *
 * The expected sizes and indices below were produced by the Python code this
 * reproduces (camera_depth_models/util/transform.py Resize.get_size and
 * torch.nn.functional.interpolate(mode="nearest") on CUDA), not derived by hand.
 *
 * No GPU, no dataset.
 */
#include "cdm/preprocessing.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cdm;

namespace
{
int failures = 0;
void check(bool condition, const std::string &what)
{
    if(!condition) { std::cerr << "  FAIL: " << what << std::endl; ++failures; }
}
bool near(float a, float b) { return std::fabs(a - b) <= 1e-6f * std::max(1.f, std::fabs(b)); }

template <typename F>
bool throws(F f)
{
    try { f(); } catch(const std::exception&) { return true; }
    return false;
}
} // namespace

int main()
{
    // --- network size: Resize(lower_bound, keep_aspect_ratio, multiple of 14) at 518
    struct SizeCase { int w, h, net_w, net_h; };
    const SizeCase sizes[] = {
        {640, 480, 686, 518}, {848, 480, 910, 518}, {1280, 720, 924, 518},
        {1920, 1080, 924, 518}, {480, 640, 518, 686}, {518, 518, 518, 518},
        {1000, 500, 1036, 518}, {1036, 518, 1036, 518}, {700, 500, 728, 518},
    };
    for(const SizeCase &c : sizes)
    {
        const cv::Size net = NetworkSize(cv::Size(c.w, c.h), 518);
        check(net == cv::Size(c.net_w, c.net_h),
              "network size for " + std::to_string(c.w) + "x" + std::to_string(c.h) + " is " +
              std::to_string(net.width) + "x" + std::to_string(net.height));
    }

    // --- PyTorch nearest indices
    {
        long sum = 0;
        for(int i = 0; i < 640; i++) sum += TorchNearestIndex(i, 686, 640);
        check(sum == 218858, "686->640 index sum");
        check(TorchNearestIndex(320, 686, 640) == 343, "686->640 index at 320 (float32 boundary)");
        sum = 0;
        for(int i = 0; i < 480; i++) sum += TorchNearestIndex(i, 518, 480);
        check(sum == 123822, "518->480 index sum");
        const int down[] = {0, 1, 2, 4, 5};
        for(int i = 0; i < 5; i++) check(TorchNearestIndex(i, 7, 5) == down[i], "7->5 index");
        const int up[] = {0, 0, 1, 2, 2, 3, 4};
        for(int i = 0; i < 7; i++) check(TorchNearestIndex(i, 5, 7) == up[i], "5->7 index");
        for(int i = 0; i < 9; i++) check(TorchNearestIndex(i, 3, 9) == i / 3, "3->9 index");
    }

    // --- preprocessing at the network size itself, so resizing is the identity
    {
        const cv::Size size(28, 14);
        cv::Mat bgr(size, CV_8UC3, cv::Scalar(255, 0, 0)); // pure blue
        cv::Mat depth(size, CV_32FC1, cv::Scalar(2.f));
        depth.at<float>(0, 1) = 0.f;
        depth.at<float>(0, 2) = std::numeric_limits<float>::quiet_NaN();
        depth.at<float>(0, 3) = -1.f;
        depth.at<float>(0, 4) = std::numeric_limits<float>::infinity();
        depth.at<float>(0, 5) = 0.25f;

        std::vector<float> input(4 * size.area(), -99.f);
        Preprocess(bgr, depth, size, input.data());
        const size_t area = size.area();
        check(near(input[0], (0.f - 0.485f) / 0.229f), "channel 0 is normalized red");
        check(near(input[area], (0.f - 0.456f) / 0.224f), "channel 1 is normalized green");
        check(near(input[2 * area], (1.f - 0.406f) / 0.225f), "channel 2 is normalized blue");
        const float *inv = input.data() + 3 * area;
        check(near(inv[0], 0.5f), "inverse depth of 2 m");
        check(inv[1] == 0.f, "zero depth -> zero inverse depth");
        check(inv[2] == 0.f, "NaN depth -> zero inverse depth");
        check(inv[3] == 0.f, "negative depth -> zero inverse depth");
        check(inv[4] == 0.f, "infinite depth -> zero inverse depth");
        check(near(inv[5], 4.f), "inverse depth of 0.25 m");

        check(throws([&] { Preprocess(bgr, depth, cv::Size(20, 14), input.data()); }),
              "network size not a multiple of 14 is rejected");
        cv::Mat depth16(size, CV_16UC1, cv::Scalar(1000));
        check(throws([&] { Preprocess(bgr, depth16, size, input.data()); }),
              "raw uint16 depth is rejected");
        check(throws([&] { Preprocess(bgr, cv::Mat(cv::Size(14, 14), CV_32FC1), size, input.data()); }),
              "colour/depth size mismatch is rejected");
    }

    // --- postprocessing
    {
        const cv::Size net(3, 2);
        const float inverse[6] = {0.5f, 0.f, 2.f, 1.f, 4.f, 0.25f};
        int invalid = -1;
        const cv::Mat same = Postprocess(inverse, net, net, &invalid);
        check(near(same.at<float>(0, 0), 2.f) && near(same.at<float>(0, 2), 0.5f) &&
              near(same.at<float>(1, 2), 4.f), "depth is 1 / inverse depth");
        check(same.at<float>(0, 1) == 0.f, "zero inverse depth -> no depth (0), not inf");
        check(invalid == 1, "invalid pixels are counted");

        const cv::Mat up = Postprocess(inverse, net, cv::Size(9, 4), &invalid);
        check(up.at<float>(3, 8) == same.at<float>(1, 2), "upsampling is nearest");
        check(invalid == 3 * 2, "invalid count follows the upsampled pixels");

        const float broken[6] = {0.5f, std::numeric_limits<float>::quiet_NaN(), 1.f, 1.f, 1.f, 1.f};
        check(throws([&] { Postprocess(broken, net, net); }), "non-finite network output throws");
    }

    if(failures == 0) std::cout << "test_preprocessing: OK" << std::endl;
    return failures == 0 ? 0 : 1;
}
