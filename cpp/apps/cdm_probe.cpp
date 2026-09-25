/*
* cdm_probe: check a CDM TensorRT engine against the Python reference that
* scripts/export_onnx.py writes, and time it.
*
* Three stages are compared separately, so a failure says where it comes from:
*   1. preprocessing   C++ Preprocess          vs the tensor infer_depth fed the network
*   2. network         TensorRT on that tensor vs PyTorch's output (FP32, TF32 off)
*   3. end to end      DepthCompleter          vs infer_depth's returned depth
*
* Exits 2 when a stage exceeds its tolerance. The default tolerances are set
* against the sensor, not tuned to the engine: a D435 at 2 m is noisy at ~1-2 %
* of depth, so the engine may add a small fraction of that and no more.
*/

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>

#include "cdm/depth_completer.h"
#include "cdm/preprocessing.h"
#include "cli.h"

namespace
{

struct Tolerances
{
    double preprocess_abs;   /*!< normalized input units */
    double median_rel;       /*!< of depth */
    double p99_rel;          /*!< of depth */
    double invalid_fraction; /*!< of pixels valid in one output only */
};

std::vector<float> ReadF32(const std::string &path, size_t expected)
{
    std::ifstream in(path.c_str(), std::ios::binary | std::ios::ate);
    if(!in.good()) throw std::runtime_error("cannot open '" + path + "'");
    const size_t bytes = static_cast<size_t>(in.tellg());
    if(bytes != expected * sizeof(float))
        throw std::runtime_error("'" + path + "' holds " + std::to_string(bytes / sizeof(float)) +
                                 " floats, expected " + std::to_string(expected) +
                                 " (reference for another resolution?)");
    std::vector<float> data(expected);
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    return data;
}

struct Comparison
{
    size_t compared = 0;
    size_t validity_mismatch = 0;
    double median = 0, p99 = 0, max = 0;
};

/*! Relative error where both are valid (finite, > 0); counts pixels valid in one only. */
Comparison CompareDepths(const float *ours, const float *reference, size_t n)
{
    std::vector<double> rel;
    rel.reserve(n);
    Comparison c;
    for(size_t i = 0; i < n; i++)
    {
        const bool a = std::isfinite(ours[i]) && ours[i] > 0.f;
        const bool b = std::isfinite(reference[i]) && reference[i] > 0.f;
        if(a != b) { c.validity_mismatch++; continue; }
        if(!a) continue;
        rel.push_back(std::fabs(ours[i] - reference[i]) / reference[i]);
    }
    c.compared = rel.size();
    c.median = cdm::Percentile(rel, 0.5);
    c.p99 = cdm::Percentile(rel, 0.99);
    c.max = rel.empty() ? std::nan("") : *std::max_element(rel.begin(), rel.end());
    return c;
}

bool Report(const std::string &stage, const Comparison &c, size_t n, const Tolerances &tol)
{
    const bool ok = c.compared > 0 && c.median <= tol.median_rel && c.p99 <= tol.p99_rel &&
                    c.validity_mismatch <= tol.invalid_fraction * n;
    std::cout << "[cdm_probe] " << stage << ": relative error median " << c.median
              << " p99 " << c.p99 << " max " << c.max << " over " << c.compared
              << " px; validity mismatch " << c.validity_mismatch << " px -> "
              << (ok ? "OK" : "FAIL") << std::endl;
    return ok;
}

void Usage(const char *argv0)
{
    std::cerr << "Usage: " << argv0 << " --engine E --reference DIR [--input-size 518] [--iterations 20]\n"
              << "         [--max-preprocess-abs 1e-4] [--max-median-rel 0.005] [--max-p99-rel 0.02]\n"
              << "         [--max-invalid-fraction 0.001]\n\n"
              << "DIR is the --reference-dir of scripts/export_onnx.py (docker/build_engine.sh\n"
              << "writes models/cdm_<camera>_<W>x<H>_reference).\n";
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const cdm::Cli cli(argc, argv,
                           {"engine", "reference", "input-size", "iterations", "max-preprocess-abs",
                            "max-median-rel", "max-p99-rel", "max-invalid-fraction"},
                           {"help"});
        if(cli.Has("help") || !cli.Positional().empty()) { Usage(argv[0]); return 1; }
        const Tolerances tol = {cli.Num("max-preprocess-abs", 1e-4), cli.Num("max-median-rel", 0.005),
                                cli.Num("max-p99-rel", 0.02), cli.Num("max-invalid-fraction", 0.001)};
        const int iterations = cli.Int("iterations", 20);
        if(iterations < 1) throw std::invalid_argument("--iterations must be >= 1");
        const std::string ref = cli.Required("reference");

        size_t free_before = 0, total = 0;
        cudaMemGetInfo(&free_before, &total);
        cdm::DepthCompleter completer(cli.Required("engine"), cli.Int("input-size", 518));
        std::cout << "[cdm_probe] " << completer.Describe() << std::endl;

        const cv::Mat bgr = cv::imread(ref + "/rgb.png", cv::IMREAD_COLOR);
        if(bgr.empty()) throw std::runtime_error("cannot read " + ref + "/rgb.png");
        const cv::Size image = bgr.size();
        const cv::Size net = completer.NetworkSize();
        const size_t image_px = image.area();
        const size_t net_px = net.area();

        std::vector<float> depth_data = ReadF32(ref + "/depth_m.f32", image_px);
        const cv::Mat depth_m(image, CV_32FC1, depth_data.data());
        const std::vector<float> ref_input = ReadF32(ref + "/input_rgbd.f32", 4 * net_px);
        const std::vector<float> ref_inverse = ReadF32(ref + "/inverse_depth.f32", net_px);
        const std::vector<float> ref_depth = ReadF32(ref + "/depth.f32", image_px);

        bool ok = true;

        // 1. preprocessing
        std::vector<float> input(4 * net_px);
        cdm::Preprocess(bgr, depth_m, net, input.data());
        double rgb_max = 0, inv_max = 0;
        for(size_t i = 0; i < 3 * net_px; i++)
            rgb_max = std::max(rgb_max, static_cast<double>(std::fabs(input[i] - ref_input[i])));
        for(size_t i = 3 * net_px; i < 4 * net_px; i++)
            inv_max = std::max(inv_max, static_cast<double>(std::fabs(input[i] - ref_input[i])));
        const bool pre_ok = rgb_max <= tol.preprocess_abs && inv_max <= tol.preprocess_abs;
        std::cout << "[cdm_probe] preprocessing: max |diff| rgb " << rgb_max << " inverse depth "
                  << inv_max << " -> " << (pre_ok ? "OK" : "FAIL") << std::endl;
        ok = ok && pre_ok;

        // 2. network alone, on the reference's own input, compared as depths.
        std::vector<float> inverse(net_px);
        completer.InferNetwork(ref_input.data(), inverse.data());
        std::vector<float> net_depth(net_px), ref_net_depth(net_px);
        for(size_t i = 0; i < net_px; i++)
        {
            net_depth[i] = inverse[i] > 0.f ? 1.f / inverse[i] : 0.f;
            ref_net_depth[i] = ref_inverse[i] > 0.f ? 1.f / ref_inverse[i] : 0.f;
        }
        ok = Report("network", CompareDepths(net_depth.data(), ref_net_depth.data(), net_px),
                    net_px, tol) && ok;

        // 3. end to end
        const cv::Mat completed = completer.Complete(bgr, depth_m);
        ok = Report("end to end", CompareDepths(completed.ptr<float>(), ref_depth.data(), image_px),
                    image_px, tol) && ok;

        size_t free_after = 0;
        cudaMemGetInfo(&free_after, &total);
        std::cout << "[cdm_probe] GPU memory taken: " << (free_before - free_after) / (1024.0 * 1024.0)
                  << " MiB" << std::endl;

        // Latency: the calls above were warm-up.
        std::vector<double> series[6];
        for(int i = 0; i < iterations; i++)
        {
            completer.Complete(bgr, depth_m);
            const cdm::Timings &t = completer.LastTimings();
            const double v[6] = {t.preprocess_ms, t.upload_ms, t.inference_ms, t.download_ms,
                                 t.postprocess_ms, t.total_ms};
            for(int s = 0; s < 6; s++) series[s].push_back(v[s]);
        }
        const char *names[] = {"preprocess (CPU)", "upload (GPU)", "inference (GPU)",
                               "download (GPU)", "postprocess (CPU)", "total (wall)"};
        std::cout << "[cdm_probe] latency over " << iterations << " calls, ms (mean / median / p95):\n";
        for(int s = 0; s < 6; s++)
            std::cout << "          " << names[s] << ": " << cdm::Mean(series[s]) << " / "
                      << cdm::Percentile(series[s], 0.5) << " / " << cdm::Percentile(series[s], 0.95) << "\n";

        std::cout << "[cdm_probe] " << (ok ? "PASS" : "FAIL") << std::endl;
        return ok ? 0 : 2;
    }
    catch(const std::invalid_argument &e)
    {
        std::cerr << "[cdm_probe] ERROR: " << e.what() << std::endl;
        Usage(argv[0]);
        return 1;
    }
    catch(const std::exception &e)
    {
        std::cerr << "[cdm_probe] ERROR: " << e.what() << std::endl;
        return 1;
    }
}
