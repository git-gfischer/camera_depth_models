/*
* CDM (camera depth model) on TensorRT.
*
* Runs an engine built by docker/build_engine.sh: an RGB image and the sensor's
* noisy, holed depth in, dense metric depth out. Pre- and post-processing are
* in preprocessing.h.
*
* The output is the network's depth everywhere, not the sensor depth with its
* holes filled; which pixels to trust from it is the caller's decision.
*
* Not thread-safe: one instance per thread that calls it.
*/

#ifndef CDM_DEPTH_COMPLETER_H
#define CDM_DEPTH_COMPLETER_H

#include <memory>
#include <string>

#include <opencv2/core/core.hpp>

namespace cdm
{

/*! Wall-clock and GPU time of the last Complete() call, in milliseconds. */
struct Timings
{
    double preprocess_ms = 0.0;  /*!< CPU */
    double upload_ms = 0.0;      /*!< GPU, host-to-device copy */
    double inference_ms = 0.0;   /*!< GPU, the engine */
    double download_ms = 0.0;    /*!< GPU, device-to-host copy */
    double postprocess_ms = 0.0; /*!< CPU */
    double total_ms = 0.0;       /*!< wall clock of the whole call */
};

class DepthCompleter
{
public:
    /*!
     * engine_path  serialized engine.
     * input_size   infer_depth's input_size the engine was exported with. It
     *              fixes the network resolution for a camera resolution; the
     *              engine is checked against it on the first frame.
     */
    DepthCompleter(const std::string &engine_path, int input_size);
    ~DepthCompleter();

    DepthCompleter(const DepthCompleter&) = delete;
    DepthCompleter& operator=(const DepthCompleter&) = delete;

    /*!
     * bgr     CV_8UC3.
     * depth_m CV_32FC1 metres, same size; non-finite or <= 0 is missing.
     * Returns CV_32FC1 metres at the input size, 0 where the network predicts
     * no depth. Throws if the image size is not the one the engine was built
     * for, or if the engine produces non-finite values.
     */
    cv::Mat Complete(const cv::Mat &bgr, const cv::Mat &depth_m);

    /*!
     * The engine alone, for parity checks: `input` holds 4*h*w floats (the
     * layout Preprocess writes), `output` receives h*w inverse depths.
     */
    void InferNetwork(const float *input, float *output);

    cv::Size NetworkSize() const;
    const Timings& LastTimings() const { return mTimings; }
    std::string Describe() const;

private:
    class Engine; /*!< TensorRT runtime wrapper, defined in the .cpp */

    std::string mEnginePath;
    int mInputSize;
    std::unique_ptr<Engine> mpEngine;
    cv::Size mCheckedImageSize;
    Timings mTimings;
};

} // namespace cdm

#endif
