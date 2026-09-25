/*
* CDM apps: camera intrinsics as camera.yaml.
*
* CDM does not use intrinsics: the network sees images only. They are carried
* alongside the depth so that whoever back-projects it later (a mapper, a point
* cloud) has the calibration of the camera that took it. cdm_realsense --record
* writes one; a dataset config names one (or holds it inline) and cdm_dataset
* checks it against the frames and copies it into the output.
*
*   width: 640
*   height: 480
*   fx: 614.317
*   fy: 614.949
*   cx: 315.525
*   cy: 230.055
*   distortion_model: plumb_bob          # OpenCV radial-tangential
*   distortion: [k1, k2, p1, p2, k3]
*   depth_factor: 1000                   # optional: raw depth units per metre
*/

#ifndef CDM_APPS_CAMERA_H
#define CDM_APPS_CAMERA_H

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace cdm
{

struct Intrinsics
{
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::string distortion_model = "none";
    std::vector<double> distortion;
    double depth_factor = 0; /*!< 0 = not stated */
};

/*! `node` is a camera map, or a string path (relative to `base`) to a camera.yaml. */
inline Intrinsics LoadIntrinsics(const YAML::Node &node, const std::filesystem::path &base)
{
    YAML::Node map = node;
    std::string where = "camera";
    if(node.IsScalar())
    {
        std::filesystem::path path(node.as<std::string>());
        if(!path.is_absolute()) path = base / path;
        where = path.string();
        try { map = YAML::LoadFile(path.string()); }
        catch(const YAML::Exception &e) { throw std::invalid_argument("cannot read " + where + ": " + e.what()); }
    }
    if(!map.IsMap()) throw std::invalid_argument(where + " must be a map of intrinsics");

    Intrinsics K;
    try
    {
        for(const auto &entry : map)
        {
            const std::string key = entry.first.as<std::string>();
            const YAML::Node &v = entry.second;
            if(key == "width") K.width = v.as<int>();
            else if(key == "height") K.height = v.as<int>();
            else if(key == "fx") K.fx = v.as<double>();
            else if(key == "fy") K.fy = v.as<double>();
            else if(key == "cx") K.cx = v.as<double>();
            else if(key == "cy") K.cy = v.as<double>();
            else if(key == "distortion_model") K.distortion_model = v.as<std::string>();
            else if(key == "distortion") K.distortion = v.as<std::vector<double>>();
            else if(key == "depth_factor") K.depth_factor = v.as<double>();
            else throw std::invalid_argument(where + ": unknown key '" + key + "'");
        }
    }
    catch(const YAML::Exception &e) { throw std::invalid_argument(where + ": " + e.what()); }

    if(K.width <= 0 || K.height <= 0 || !(K.fx > 0) || !(K.fy > 0))
        throw std::invalid_argument(where + ": width, height, fx and fy are required and positive");
    if(!(K.cx > 0 && K.cx < K.width && K.cy > 0 && K.cy < K.height))
        throw std::invalid_argument(where + ": the principal point lies outside the image");
    if(K.depth_factor < 0) throw std::invalid_argument(where + ": depth_factor must be positive");
    return K;
}

inline void SaveIntrinsics(const Intrinsics &K, const std::filesystem::path &path, const std::string &comment)
{
    std::ofstream out(path);
    if(!out.good()) throw std::runtime_error("cannot write " + path.string());
    out << "# " << comment << "\n" << std::setprecision(12)
        << "width: " << K.width << "\nheight: " << K.height
        << "\nfx: " << K.fx << "\nfy: " << K.fy << "\ncx: " << K.cx << "\ncy: " << K.cy
        << "\ndistortion_model: " << K.distortion_model << "\ndistortion: [";
    for(size_t i = 0; i < K.distortion.size(); i++) out << (i ? ", " : "") << K.distortion[i];
    out << "]\n";
    if(K.depth_factor > 0) out << "depth_factor: " << K.depth_factor << "\n";
}

} // namespace cdm

#endif
