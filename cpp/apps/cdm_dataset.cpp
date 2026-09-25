/*
* cdm_dataset: run CDM over a recorded RGB-D sequence.
*
* Input, one of:
*   --associations FILE   TUM format, "rgb_ts rgb_path depth_ts depth_path" per
*                         line, paths relative to the file's directory
*                         (cdm_realsense --record writes one);
*   --rgb-dir D --depth-dir D   frames paired by identical file name.
*
* Output directory:
*   depth/<depth file name>   CDM depth, uint16 PNG at the input's depth factor
*   associations.txt          TUM format, the input RGB (absolute path, as this
*                             process sees it) with the CDM depth, so the output
*                             replays as a sequence of its own
*   viz/<name>.jpg            RGB | sensor | CDM, with --viz
*   timings.csv               per-frame latency (ms)
*   camera.yaml               the dataset's intrinsics, when the config gives them
*   run.yaml                  every setting in effect
*
* Settings come from --config FILE (config/dataset.yaml) and/or the command line.
*/

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cdm/depth_completer.h"
#include "cdm/io.h"
#include "camera.h"
#include "cli.h"

namespace fs = std::filesystem;

namespace
{

struct Frame
{
    std::string rgb_ts, depth_ts; // empty for directory input
    fs::path rgb, depth;
};

std::vector<Frame> ReadAssociations(const fs::path &file)
{
    std::ifstream in(file);
    if(!in.good()) throw std::runtime_error("cannot open " + file.string());
    const fs::path base = file.parent_path();
    std::vector<Frame> frames;
    std::string line;
    int line_no = 0;
    while(std::getline(in, line))
    {
        line_no++;
        if(line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        Frame f;
        std::string rgb, depth;
        if(!(fields >> f.rgb_ts >> rgb >> f.depth_ts >> depth))
            throw std::runtime_error(file.string() + ":" + std::to_string(line_no) +
                                     ": expected 'rgb_ts rgb_path depth_ts depth_path'");
        f.rgb = base / rgb;
        f.depth = base / depth;
        frames.push_back(f);
    }
    return frames;
}

std::vector<Frame> PairDirectories(const fs::path &rgb_dir, const fs::path &depth_dir)
{
    std::map<std::string, fs::path> depth_by_name;
    for(const auto &entry : fs::directory_iterator(depth_dir))
        if(entry.is_regular_file()) depth_by_name[entry.path().filename().string()] = entry.path();
    std::vector<Frame> frames;
    size_t rgb_count = 0;
    std::map<std::string, fs::path> rgb_sorted;
    for(const auto &entry : fs::directory_iterator(rgb_dir))
        if(entry.is_regular_file()) rgb_sorted[entry.path().filename().string()] = entry.path();
    for(const auto &kv : rgb_sorted)
    {
        rgb_count++;
        const auto it = depth_by_name.find(kv.first);
        if(it == depth_by_name.end()) continue;
        Frame f;
        f.rgb = kv.second;
        f.depth = it->second;
        f.rgb_ts = f.depth_ts = kv.second.stem().string();
        frames.push_back(f);
    }
    std::cout << "[cdm_dataset] paired " << frames.size() << " frames by name ("
              << rgb_count - frames.size() << " rgb and " << depth_by_name.size() - frames.size()
              << " depth files unpaired)" << std::endl;
    return frames;
}

void Usage(const char *argv0)
{
    std::cerr
        << "Usage: " << argv0 << " [--config dataset.yaml] --engine E --output DIR\n"
        << "         (--associations FILE | --rgb-dir D --depth-dir D)\n"
        << "         [--depth-factor 1000] [--input-size 518] [--stride 1] [--max-frames 0]\n"
        << "         [--viz] [--show] [--max-depth-viz M]\n\n"
        << "  --config FILE    YAML with these settings (keys with '_'), engine_config and camera;\n"
        << "                   options on the command line override it\n"
        << "  --depth-factor   raw depth units per metre of the input PNGs; the output uses the same\n"
        << "                   (default: the camera's depth_factor, else 1000)\n"
        << "  --stride N       process every N-th frame (e.g. to mimic keyframes)\n"
        << "  --max-frames N   stop after N processed frames (0 = all)\n"
        << "  --viz            write RGB | sensor | CDM previews to <output>/viz\n"
        << "  --show           display the previews while running\n"
        << "  --max-depth-viz  colour range of the previews in metres (default: from the first frame)\n";
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const cdm::Cli cli(argc, argv,
                           {"engine", "output", "associations", "rgb-dir", "depth-dir", "depth-factor",
                            "input-size", "stride", "max-frames", "max-depth-viz"},
                           {"viz", "show", "help"},
                           {"engine", "output", "associations", "rgb-dir", "depth-dir"},
                           /*accepts_camera=*/true);
        if(cli.Has("help") || !cli.Positional().empty()) { Usage(argv[0]); return 1; }

        // Intrinsics are not used by CDM; they are checked and passed through.
        cdm::Intrinsics K;
        const bool have_camera = cli.Camera().IsDefined() && !cli.Camera().IsNull();
        if(have_camera) K = cdm::LoadIntrinsics(cli.Camera(), cli.CameraBase());
        if(have_camera && K.depth_factor > 0 && cli.Has("depth-factor") &&
           cli.Num("depth-factor", 0) != K.depth_factor)
            throw std::invalid_argument("depth_factor " + cli.Str("depth-factor", "") +
                                        " contradicts the camera's depth_factor");
        const double depth_factor =
            cli.Num("depth-factor", (have_camera && K.depth_factor > 0) ? K.depth_factor : 1000.0);
        K.depth_factor = depth_factor;
        const int stride = cli.Int("stride", 1);
        const int max_frames = cli.Int("max-frames", 0);
        float max_viz = static_cast<float>(cli.Num("max-depth-viz", 0.0));
        const bool viz = cli.Has("viz"), show = cli.Has("show");
        if(stride < 1 || max_frames < 0) throw std::invalid_argument("--stride >= 1, --max-frames >= 0");

        std::vector<Frame> frames;
        if(cli.Has("associations"))
        {
            if(cli.Has("rgb-dir") || cli.Has("depth-dir"))
                throw std::invalid_argument("give --associations or --rgb-dir/--depth-dir, not both");
            frames = ReadAssociations(cli.Required("associations"));
        }
        else
            frames = PairDirectories(cli.Required("rgb-dir"), cli.Required("depth-dir"));
        if(frames.empty()) throw std::runtime_error("no frames to process");

        const fs::path out = cli.Required("output");
        fs::create_directories(out / "depth");
        if(viz) fs::create_directories(out / "viz");

        cdm::DepthCompleter completer(cli.Required("engine"), cli.Int("input-size", 518));
        std::cout << "[cdm_dataset] " << completer.Describe() << " | " << frames.size()
                  << " frames, stride " << stride << std::endl;

        {
            // A dataset config of its own: `--config <output>/run.yaml` repeats the run.
            std::ofstream run(out / "run.yaml");
            run << "# cdm_dataset settings in effect" << (cli.ConfigPath().empty() ? "" : " (config " +
                   cli.ConfigPath() + ")") << "; usable as --config\n";
            std::map<std::string, std::string> effective = cli.Effective();
            effective["depth-factor"] = std::to_string(depth_factor);
            for(const auto &kv : effective)
            {
                std::string key = kv.first;
                std::replace(key.begin(), key.end(), '-', '_');
                run << key << ": \"" << kv.second << "\"\n";
            }
            if(have_camera) run << "camera: camera.yaml\n";
        }
        if(have_camera)
            cdm::SaveIntrinsics(K, out / "camera.yaml", "intrinsics of the input camera; CDM depth is in its frame");

        std::ofstream assoc(out / "associations.txt");
        assoc << "# rgb_ts rgb depth_ts cdm_depth (CDM depth, depth factor " << depth_factor << ")\n";
        std::ofstream csv(out / "timings.csv");
        csv << "frame,preprocess_ms,upload_ms,inference_ms,download_ms,postprocess_ms,total_ms,"
               "invalid_px,clipped_px\n";

        std::vector<double> totals, inference;
        long invalid_total = 0, clipped_total = 0, pixels_total = 0;
        int processed = 0;
        for(size_t i = 0; i < frames.size(); i += stride)
        {
            if(max_frames > 0 && processed >= max_frames) break;
            const Frame &f = frames[i];
            const cv::Mat bgr = cv::imread(f.rgb.string(), cv::IMREAD_COLOR);
            if(bgr.empty()) throw std::runtime_error("cannot read " + f.rgb.string());
            const cv::Mat depth_m = cdm::LoadDepthMetres(f.depth.string(), depth_factor);
            if(have_camera && processed == 0 && (bgr.cols != K.width || bgr.rows != K.height))
                throw std::invalid_argument("the camera is " + std::to_string(K.width) + "x" +
                                            std::to_string(K.height) + " but the frames are " +
                                            std::to_string(bgr.cols) + "x" + std::to_string(bgr.rows));

            const cv::Mat completed = completer.Complete(bgr, depth_m);
            const cdm::Timings t = completer.LastTimings();

            int clipped = 0;
            const cv::Mat out_u16 = cdm::DepthToU16(completed, depth_factor, &clipped);
            const int invalid = static_cast<int>(completed.total()) - cv::countNonZero(completed);
            const fs::path depth_out = fs::path("depth") / f.depth.filename().replace_extension(".png");
            if(!cv::imwrite((out / depth_out).string(), out_u16))
                throw std::runtime_error("cannot write " + (out / depth_out).string());
            assoc << f.rgb_ts << " " << fs::absolute(f.rgb).string() << " " << f.depth_ts << " "
                  << depth_out.string() << "\n";
            csv << f.depth.filename().string() << std::fixed << std::setprecision(3) << ","
                << t.preprocess_ms << "," << t.upload_ms << "," << t.inference_ms << ","
                << t.download_ms << "," << t.postprocess_ms << "," << t.total_ms << ","
                << invalid << "," << clipped << "\n";

            if(viz || show)
            {
                if(max_viz <= 0.f) max_viz = cdm::AutoDepthRange(depth_m);
                std::ostringstream status;
                status << std::fixed << std::setprecision(1) << "frame " << i << "  " << t.total_ms << " ms";
                const cv::Mat panel = cdm::SideBySide(bgr, depth_m, completed, max_viz, status.str());
                if(viz)
                    cv::imwrite((out / "viz" / f.rgb.filename().replace_extension(".jpg")).string(), panel);
                if(show)
                {
                    cv::imshow("cdm_dataset", panel);
                    const int key = cv::waitKey(1);
                    if(key == 'q' || key == 27) break;
                }
            }

            // The first calls include TensorRT's lazy initialisation.
            if(processed >= 3) { totals.push_back(t.total_ms); inference.push_back(t.inference_ms); }
            invalid_total += invalid;
            clipped_total += clipped;
            pixels_total += static_cast<long>(completed.total());
            processed++;
            if(processed % 100 == 0)
                std::cout << "[cdm_dataset] " << processed << " frames" << std::endl;
        }

        std::cout << std::fixed << std::setprecision(2)
                  << "[cdm_dataset] done: " << processed << " frames -> " << out.string() << "\n"
                  << "  latency ms (excluding 3 warm-up frames): total mean " << cdm::Mean(totals)
                  << " / median " << cdm::Percentile(totals, 0.5) << " / p95 "
                  << cdm::Percentile(totals, 0.95) << "; inference median "
                  << cdm::Percentile(inference, 0.5) << "\n"
                  << "  pixels without CDM depth: " << invalid_total << " of " << pixels_total
                  << "; beyond the uint16 range (written as 0): " << clipped_total << std::endl;
        return 0;
    }
    catch(const std::invalid_argument &e)
    {
        std::cerr << "[cdm_dataset] ERROR: " << e.what() << std::endl;
        Usage(argv[0]);
        return 1;
    }
    catch(const std::exception &e)
    {
        std::cerr << "[cdm_dataset] ERROR: " << e.what() << std::endl;
        return 1;
    }
}
