/*
* cdm_realsense: run CDM live on an Intel RealSense camera.
*
* Colour and depth are streamed at the same resolution and the depth is
* aligned to the colour camera (rs2::align), which is what infer_depth expects:
* one RGB and one depth image of the same view.
*
* With --record DIR every processed frame is written as a TUM-style sequence
* that cdm_dataset replays:
*   rgb/<ts>.png, depth/<ts>.png   colour and aligned raw sensor depth
*   cdm/<ts>.png                   CDM depth, same depth factor as depth/
*   associations.txt               "ts rgb/<ts>.png ts depth/<ts>.png"
*   camera.yaml                    the device's colour intrinsics and depth factor,
*                                  usable as `camera:` in a dataset config
*
* Settings come from --config FILE (config/realsense.yaml, whose engine_config
* also sets the stream resolution) and/or the command line.
*/

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cdm/depth_completer.h"
#include "cdm/io.h"
#include "camera.h"
#include "cli.h"

namespace fs = std::filesystem;

namespace
{

std::atomic<bool> gStop(false);
void OnSignal(int) { gStop = true; }

void Usage(const char *argv0)
{
    std::cerr
        << "Usage: " << argv0 << " [--config realsense.yaml] --engine E [--input-size 518]\n"
        << "         [--width 640] [--height 480] [--fps 30] [--serial S]\n"
        << "         [--warmup 30] [--frames 0] [--record DIR] [--no-display] [--max-depth-viz M]\n"
        << "       " << argv0 << " --list\n\n"
        << "  --config FILE     YAML with these settings (keys with '_') and engine_config;\n"
        << "                    options on the command line override it\n"
        << "  --width/--height  stream resolution; must be the one the engine was built for\n"
        << "  --warmup N        frames skipped at start while auto-exposure settles\n"
        << "  --frames N        stop after N processed frames (0 = until q/ESC or Ctrl-C)\n"
        << "  --record DIR      save rgb, raw depth and CDM depth as a TUM-style sequence\n"
        << "  --no-display      no window; print one line of statistics per second\n"
        << "  --max-depth-viz M colour range in metres (default: from the first frame)\n";
}

int ListDevices()
{
    rs2::context ctx;
    const rs2::device_list devices = ctx.query_devices();
    if(devices.size() == 0) { std::cout << "no RealSense device found" << std::endl; return 1; }
    for(const rs2::device &d : devices)
        std::cout << d.get_info(RS2_CAMERA_INFO_NAME) << "  serial "
                  << d.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << "  firmware "
                  << d.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << std::endl;
    return 0;
}

std::string Stamp(double ms)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(6) << ms / 1000.0;
    return s.str();
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const cdm::Cli cli(argc, argv,
                           {"engine", "input-size", "width", "height", "fps", "serial", "warmup",
                            "frames", "record", "max-depth-viz"},
                           {"no-display", "list", "help"},
                           {"engine", "record"});
        if(cli.Has("help") || !cli.Positional().empty()) { Usage(argv[0]); return 1; }
        if(cli.Has("list")) return ListDevices();

        const int width = cli.Int("width", 640), height = cli.Int("height", 480), fps = cli.Int("fps", 30);
        const int warmup = cli.Int("warmup", 30), max_frames = cli.Int("frames", 0);
        float max_viz = static_cast<float>(cli.Num("max-depth-viz", 0.0));
        const bool display = !cli.Has("no-display");
        const std::string record = cli.Str("record", "");

        // Load the engine first: a mismatch should fail before the camera starts.
        cdm::DepthCompleter completer(cli.Required("engine"), cli.Int("input-size", 518));
        std::cout << "[cdm_realsense] " << completer.Describe() << std::endl;

        rs2::config config;
        if(cli.Has("serial")) config.enable_device(cli.Required("serial"));
        config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
        config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
        rs2::pipeline pipe;
        const rs2::pipeline_profile profile = pipe.start(config);
        const rs2::device device = profile.get_device();
        const float depth_scale = device.first<rs2::depth_sensor>().get_depth_scale();
        const double depth_factor = 1.0 / depth_scale;
        const rs2_intrinsics K =
            profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
        std::cout << "[cdm_realsense] " << device.get_info(RS2_CAMERA_INFO_NAME) << " serial "
                  << device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << " | " << width << "x" << height
                  << "@" << fps << " | depth factor " << depth_factor << " units/m" << std::endl;

        std::ofstream assoc;
        if(!record.empty())
        {
            for(const char *sub : {"rgb", "depth", "cdm"}) fs::create_directories(fs::path(record) / sub);
            assoc.open(fs::path(record) / "associations.txt");
            assoc << "# rgb_ts rgb depth_ts depth (depth aligned to colour, factor " << depth_factor << ")\n";
            cdm::Intrinsics intrinsics;
            intrinsics.width = K.width;
            intrinsics.height = K.height;
            intrinsics.fx = K.fx;
            intrinsics.fy = K.fy;
            intrinsics.cx = K.ppx;
            intrinsics.cy = K.ppy;
            intrinsics.distortion_model = rs2_distortion_to_string(K.model);
            intrinsics.distortion.assign(K.coeffs, K.coeffs + 5);
            intrinsics.depth_factor = depth_factor;
            cdm::SaveIntrinsics(intrinsics, fs::path(record) / "camera.yaml",
                                std::string("factory colour intrinsics of ") +
                                device.get_info(RS2_CAMERA_INFO_NAME) + " serial " +
                                device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) +
                                "; depth is aligned to colour");
        }

        std::signal(SIGINT, OnSignal);
        std::signal(SIGTERM, OnSignal);
        rs2::align align(RS2_STREAM_COLOR);

        for(int i = 0; i < warmup && !gStop; i++) pipe.wait_for_frames();

        int processed = 0;
        std::vector<double> window_ms;
        auto window_start = std::chrono::steady_clock::now();
        while(!gStop && (max_frames == 0 || processed < max_frames))
        {
            const rs2::frameset frames = align.process(pipe.wait_for_frames());
            const rs2::video_frame colour = frames.get_color_frame();
            const rs2::depth_frame depth = frames.get_depth_frame();
            if(!colour || !depth) continue;

            const cv::Mat bgr(height, width, CV_8UC3, const_cast<void*>(colour.get_data()));
            const cv::Mat raw(height, width, CV_16UC1, const_cast<void*>(depth.get_data()));
            cv::Mat depth_m;
            raw.convertTo(depth_m, CV_32F, depth_scale);

            const cv::Mat completed = completer.Complete(bgr, depth_m);
            const cdm::Timings t = completer.LastTimings();
            processed++;
            window_ms.push_back(t.total_ms);

            if(!record.empty())
            {
                const std::string ts = Stamp(colour.get_timestamp());
                cv::imwrite((fs::path(record) / "rgb" / (ts + ".png")).string(), bgr);
                cv::imwrite((fs::path(record) / "depth" / (ts + ".png")).string(), raw);
                cv::imwrite((fs::path(record) / "cdm" / (ts + ".png")).string(),
                            cdm::DepthToU16(completed, depth_factor));
                assoc << ts << " rgb/" << ts << ".png " << ts << " depth/" << ts << ".png\n";
            }

            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - window_start).count();
            std::ostringstream status;
            status << std::fixed << std::setprecision(1) << "CDM " << t.total_ms << " ms (engine "
                   << t.inference_ms << ")";
            if(elapsed >= 1.0)
            {
                if(!display)
                    std::cout << "[cdm_realsense] " << window_ms.size() / elapsed << " fps, CDM median "
                              << cdm::Percentile(window_ms, 0.5) << " ms, engine "
                              << t.inference_ms << " ms" << std::endl;
                window_ms.clear();
                window_start = now;
            }
            if(display)
            {
                if(max_viz <= 0.f) max_viz = cdm::AutoDepthRange(depth_m);
                cv::imshow("cdm_realsense (q to quit)", cdm::SideBySide(bgr, depth_m, completed, max_viz, status.str()));
                const int key = cv::waitKey(1);
                if(key == 'q' || key == 27) break;
            }
        }
        pipe.stop();
        std::cout << "[cdm_realsense] processed " << processed << " frames"
                  << (record.empty() ? "" : ", recorded to " + record) << std::endl;
        return 0;
    }
    catch(const std::invalid_argument &e)
    {
        std::cerr << "[cdm_realsense] ERROR: " << e.what() << std::endl;
        Usage(argv[0]);
        return 1;
    }
    catch(const rs2::error &e)
    {
        std::cerr << "[cdm_realsense] RealSense error in " << e.get_failed_function() << ": "
                  << e.what() << std::endl;
        return 1;
    }
    catch(const std::exception &e)
    {
        std::cerr << "[cdm_realsense] ERROR: " << e.what() << std::endl;
        return 1;
    }
}
