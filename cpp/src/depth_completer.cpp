/* CDM on TensorRT. */

#include "cdm/depth_completer.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cdm/preprocessing.h"

namespace cdm
{

namespace {

typedef std::chrono::steady_clock Clock;

double MsSince(const Clock::time_point &start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void CheckCuda(cudaError_t status, const char *what)
{
    if(status != cudaSuccess)
        throw std::runtime_error(std::string("CDM: ") + what + ": " + cudaGetErrorString(status));
}

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if(severity <= Severity::kWARNING)
            std::cerr << "[cdm][TRT] " << msg << std::endl;
    }
};

} // namespace

/*! Deserializes the engine, owns the buffers and runs one synchronous pass. */
class DepthCompleter::Engine
{
public:
    explicit Engine(const std::string &engine_path);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    cv::Size NetworkSize() const { return mNetSize; }
    size_t InputElements() const { return static_cast<size_t>(4) * mNetSize.area(); }
    size_t OutputElements() const { return static_cast<size_t>(mNetSize.area()); }
    float* HostInput() { return mInputPinned; }
    const float* HostOutput() const { return mOutputPinned; }

    /*! Pinned input -> pinned output; fills the three GPU timings. */
    void Infer(Timings &timings);

private:
    TrtLogger mLogger;
    std::unique_ptr<nvinfer1::IRuntime> mpRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mpEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mpContext;

    cv::Size mNetSize;
    void *mInputDevice = nullptr;
    void *mOutputDevice = nullptr;
    float *mInputPinned = nullptr;
    float *mOutputPinned = nullptr;
    cudaStream_t mStream = nullptr;
    cudaEvent_t mEvents[4] = {nullptr, nullptr, nullptr, nullptr};
};

DepthCompleter::Engine::Engine(const std::string &engine_path)
{
    std::ifstream file(engine_path.c_str(), std::ios::binary | std::ios::ate);
    if(!file.good())
        throw std::runtime_error("CDM: cannot open TensorRT engine '" + engine_path + "'");
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<size_t>(size));
    if(!file.read(blob.data(), size))
        throw std::runtime_error("CDM: cannot read TensorRT engine '" + engine_path + "'");

    mpRuntime.reset(nvinfer1::createInferRuntime(mLogger));
    if(!mpRuntime) throw std::runtime_error("CDM: cannot create the TensorRT runtime");
    mpEngine.reset(mpRuntime->deserializeCudaEngine(blob.data(), blob.size()));
    if(!mpEngine)
        throw std::runtime_error("CDM: cannot deserialize TensorRT engine '" + engine_path +
                                 "' (built by another TensorRT version or for another GPU?)");
    mpContext.reset(mpEngine->createExecutionContext());
    if(!mpContext) throw std::runtime_error("CDM: cannot create TensorRT execution context");

    // Exactly the export's two tensors: rgbd [1,4,h,w] and inverse_depth [1,h,w], float32.
    if(mpEngine->getNbIOTensors() != 2)
        throw std::runtime_error("CDM: engine must have one input and one output tensor");
    const char *input_name = nullptr;
    const char *output_name = nullptr;
    for(int i = 0; i < 2; i++)
    {
        const char *name = mpEngine->getIOTensorName(i);
        if(mpEngine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error(std::string("CDM: tensor '") + name + "' is not float32");
        if(mpEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name = name;
        else output_name = name;
    }
    if(input_name == nullptr || output_name == nullptr)
        throw std::runtime_error("CDM: engine must have one input and one output tensor");

    const nvinfer1::Dims in = mpEngine->getTensorShape(input_name);
    const nvinfer1::Dims out = mpEngine->getTensorShape(output_name);
    if(in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 4 || in.d[2] <= 0 || in.d[3] <= 0)
        throw std::runtime_error("CDM: input must be a static [1,4,h,w] tensor");
    mNetSize = cv::Size(static_cast<int>(in.d[3]), static_cast<int>(in.d[2]));
    if(out.nbDims != 3 || out.d[0] != 1 || out.d[1] != in.d[2] || out.d[2] != in.d[3])
        throw std::runtime_error("CDM: output must be [1,h,w] at the input resolution");

    CheckCuda(cudaMalloc(&mInputDevice, InputElements() * sizeof(float)), "cudaMalloc input");
    CheckCuda(cudaMalloc(&mOutputDevice, OutputElements() * sizeof(float)), "cudaMalloc output");
    CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&mInputPinned), InputElements() * sizeof(float)),
              "cudaMallocHost input");
    CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&mOutputPinned), OutputElements() * sizeof(float)),
              "cudaMallocHost output");
    if(!mpContext->setTensorAddress(input_name, mInputDevice) ||
       !mpContext->setTensorAddress(output_name, mOutputDevice))
        throw std::runtime_error("CDM: cannot bind the engine tensors");

    // A blocking stream: next to other GPU work on the legacy default stream
    // (e.g. a mapper), the two take turns instead of contending for SMs.
    CheckCuda(cudaStreamCreate(&mStream), "cudaStreamCreate");
    for(cudaEvent_t &event : mEvents) CheckCuda(cudaEventCreate(&event), "cudaEventCreate");

    std::cout << "[cdm] TensorRT engine: " << engine_path
              << " | network " << mNetSize.width << "x" << mNetSize.height << std::endl;
}

DepthCompleter::Engine::~Engine()
{
    for(cudaEvent_t event : mEvents) if(event) cudaEventDestroy(event);
    if(mStream) cudaStreamDestroy(mStream);
    cudaFreeHost(mInputPinned);
    cudaFreeHost(mOutputPinned);
    cudaFree(mInputDevice);
    cudaFree(mOutputDevice);
}

void DepthCompleter::Engine::Infer(Timings &timings)
{
    CheckCuda(cudaEventRecord(mEvents[0], mStream), "cudaEventRecord");
    CheckCuda(cudaMemcpyAsync(mInputDevice, mInputPinned, InputElements() * sizeof(float),
                              cudaMemcpyHostToDevice, mStream), "input copy");
    CheckCuda(cudaEventRecord(mEvents[1], mStream), "cudaEventRecord");
    if(!mpContext->enqueueV3(mStream))
        throw std::runtime_error("CDM: TensorRT enqueue failed");
    CheckCuda(cudaEventRecord(mEvents[2], mStream), "cudaEventRecord");
    CheckCuda(cudaMemcpyAsync(mOutputPinned, mOutputDevice, OutputElements() * sizeof(float),
                              cudaMemcpyDeviceToHost, mStream), "output copy");
    CheckCuda(cudaEventRecord(mEvents[3], mStream), "cudaEventRecord");
    CheckCuda(cudaStreamSynchronize(mStream), "inference");

    float ms[3];
    for(int i = 0; i < 3; i++)
        CheckCuda(cudaEventElapsedTime(&ms[i], mEvents[i], mEvents[i + 1]), "cudaEventElapsedTime");
    timings.upload_ms = ms[0];
    timings.inference_ms = ms[1];
    timings.download_ms = ms[2];
}

DepthCompleter::DepthCompleter(const std::string &engine_path, int input_size)
    : mEnginePath(engine_path), mInputSize(input_size)
{
    if(mEnginePath.empty())
        throw std::invalid_argument("CDM: no engine path");
    if(mInputSize < kPatchSize || mInputSize % kPatchSize != 0)
        throw std::invalid_argument("CDM: input_size must be a positive multiple of 14");
    mpEngine.reset(new Engine(mEnginePath));
}

DepthCompleter::~DepthCompleter()
{
}

cv::Size DepthCompleter::NetworkSize() const
{
    return mpEngine->NetworkSize();
}

std::string DepthCompleter::Describe() const
{
    std::ostringstream out;
    out << "engine=" << mEnginePath << " network=" << NetworkSize().width << "x"
        << NetworkSize().height << " input_size=" << mInputSize;
    return out.str();
}

cv::Mat DepthCompleter::Complete(const cv::Mat &bgr, const cv::Mat &depth_m)
{
    const Clock::time_point start = Clock::now();

    // The engine has one resolution; a camera it was not built for would be
    // silently squeezed into it. Checked once per image size.
    if(bgr.size() != mCheckedImageSize)
    {
        const cv::Size expected = cdm::NetworkSize(bgr.size(), mInputSize);
        if(expected != NetworkSize())
        {
            std::ostringstream msg;
            msg << "CDM: a " << bgr.cols << "x" << bgr.rows << " image needs a "
                << expected.width << "x" << expected.height << " network at input_size "
                << mInputSize << ", but the engine is " << NetworkSize().width << "x"
                << NetworkSize().height << ". Build one with width: " << bgr.cols
                << ", height: " << bgr.rows << " in config/engine.yaml";
            throw std::runtime_error(msg.str());
        }
        mCheckedImageSize = bgr.size();
    }

    Preprocess(bgr, depth_m, NetworkSize(), mpEngine->HostInput());
    mTimings.preprocess_ms = MsSince(start);

    mpEngine->Infer(mTimings);

    const Clock::time_point post = Clock::now();
    cv::Mat depth = Postprocess(mpEngine->HostOutput(), NetworkSize(), bgr.size());
    mTimings.postprocess_ms = MsSince(post);
    mTimings.total_ms = MsSince(start);
    return depth;
}

void DepthCompleter::InferNetwork(const float *input, float *output)
{
    std::memcpy(mpEngine->HostInput(), input, mpEngine->InputElements() * sizeof(float));
    mpEngine->Infer(mTimings);
    std::memcpy(output, mpEngine->HostOutput(), mpEngine->OutputElements() * sizeof(float));
}

} // namespace cdm
