/*
* CDM apps: settings from a YAML config file and the command line.
*
* A config file (--config FILE) is a flat YAML map whose keys are the option
* names with '_' for '-' (max_frames for --max-frames). A flag is set by
* `true`; `false` and empty values leave the option unset. Two keys are
* special:
*   engine_config  path to an engine config (config/engine.yaml); its keys that
*                  the app also takes (engine, input_size, width, height) are
*                  read from it, the rest belong to the engine builder;
*   camera         intrinsics, for apps that accept them (see camera.h).
* Relative paths in a config are taken relative to that file's directory.
* Options given on the command line override the config. Unknown options and
* unknown config keys are errors, so a typo cannot silently fall back to a
* default.
*/

#ifndef CDM_APPS_CLI_H
#define CDM_APPS_CLI_H

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace cdm
{

class Cli
{
public:
    /*!
     * `options` take a value, `flags` do not; anything else is rejected.
     * `path_options` hold file paths, resolved against a config file's
     * directory when they come from one. `accepts_camera` allows a `camera`
     * config key.
     */
    Cli(int argc, char **argv, const std::set<std::string> &options, const std::set<std::string> &flags,
        const std::set<std::string> &path_options = {}, bool accepts_camera = false)
        : mOptions(options), mFlagNames(flags), mPathOptions(path_options), mAcceptsCamera(accepts_camera)
    {
        for(int i = 1; i < argc; i++)
        {
            const std::string arg = argv[i];
            if(arg.compare(0, 2, "--") != 0) { mPositional.push_back(arg); continue; }
            const std::string key = arg.substr(2);
            if(flags.count(key)) { mFlags.insert(key); continue; }
            if(key != "config" && !options.count(key)) throw std::invalid_argument("unknown option " + arg);
            if(i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
            if(key == "config") mConfigPath = argv[++i];
            else mValues[key] = argv[++i];
        }
        if(!mConfigPath.empty()) LoadConfig(mConfigPath, false);
    }

    bool Has(const std::string &key) const { return mValues.count(key) || mFlags.count(key); }
    const std::vector<std::string>& Positional() const { return mPositional; }
    const std::string& ConfigPath() const { return mConfigPath; }
    /*! The `camera` config value (a map or a path to a camera.yaml); null if absent. */
    const YAML::Node& Camera() const { return mCamera; }
    const std::filesystem::path& CameraBase() const { return mCameraBase; }
    /*! Every option and flag in effect, for recording a run's settings. */
    std::map<std::string, std::string> Effective() const
    {
        std::map<std::string, std::string> all = mValues;
        for(const std::string &f : mFlags) all[f] = "true";
        return all;
    }

    std::string Str(const std::string &key, const std::string &fallback) const
    {
        const auto it = mValues.find(key);
        return it == mValues.end() ? fallback : it->second;
    }
    std::string Required(const std::string &key) const
    {
        const auto it = mValues.find(key);
        if(it == mValues.end())
            throw std::invalid_argument("--" + key + " (config key " + ConfigKey(key) + ") is required");
        return it->second;
    }
    double Num(const std::string &key, double fallback) const
    {
        const auto it = mValues.find(key);
        if(it == mValues.end()) return fallback;
        try { return std::stod(it->second); }
        catch(...) { throw std::invalid_argument("--" + key + " is not a number: " + it->second); }
    }
    int Int(const std::string &key, int fallback) const
    {
        const double v = Num(key, fallback);
        if(v != std::floor(v)) throw std::invalid_argument("--" + key + " must be an integer");
        return static_cast<int>(v);
    }

private:
    static std::string ConfigKey(std::string key)
    {
        std::replace(key.begin(), key.end(), '-', '_');
        return key;
    }

    /*! `engine_level`: an engine config, whose keys the app does not take are the builder's. */
    void LoadConfig(const std::filesystem::path &path, bool engine_level)
    {
        YAML::Node root;
        try { root = YAML::LoadFile(path.string()); }
        catch(const YAML::Exception &e)
        {
            throw std::invalid_argument("cannot read config " + path.string() + ": " + e.what());
        }
        if(!root.IsMap()) throw std::invalid_argument("config " + path.string() + " is not a YAML map");
        const std::filesystem::path base = path.parent_path();
        std::filesystem::path engine_config;

        for(const auto &entry : root)
        {
            const std::string yaml_key = entry.first.as<std::string>();
            std::string key = yaml_key;
            std::replace(key.begin(), key.end(), '_', '-');
            const YAML::Node &value = entry.second;
            const std::string where = path.string() + ": " + yaml_key;

            if(!engine_level && key == "engine-config")
            {
                engine_config = Resolve(base, value.as<std::string>());
                continue;
            }
            if(!engine_level && key == "camera")
            {
                if(!mAcceptsCamera) throw std::invalid_argument(where + ": this tool takes no intrinsics");
                mCamera = value;
                mCameraBase = base;
                continue;
            }
            if(mFlagNames.count(key))
            {
                bool on = false;
                try { on = value.as<bool>(); }
                catch(const YAML::Exception&) { throw std::invalid_argument(where + " must be true or false"); }
                if(on) mFlags.insert(key);
                continue;
            }
            if(mOptions.count(key))
            {
                if(!value.IsScalar()) throw std::invalid_argument(where + " must be a single value");
                const std::string text = value.as<std::string>();
                if(text.empty() || mValues.count(key)) continue; // unset, or given on the command line
                mValues[key] = mPathOptions.count(key) ? Resolve(base, text).string() : text;
                continue;
            }
            if(engine_level) continue;
            throw std::invalid_argument(where + " is not a setting of this tool");
        }
        // Last, so this file's own keys win over the engine config's.
        if(!engine_config.empty()) LoadConfig(engine_config, true);
    }

    static std::filesystem::path Resolve(const std::filesystem::path &base, const std::string &p)
    {
        const std::filesystem::path path(p);
        return path.is_absolute() ? path : base / path;
    }

    std::set<std::string> mOptions, mFlagNames, mPathOptions;
    bool mAcceptsCamera;
    std::map<std::string, std::string> mValues;
    std::set<std::string> mFlags;
    std::vector<std::string> mPositional;
    std::string mConfigPath;
    YAML::Node mCamera;
    std::filesystem::path mCameraBase;
};

inline double Percentile(std::vector<double> v, double q)
{
    if(v.empty()) return std::nan("");
    const size_t k = std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

inline double Mean(const std::vector<double> &v)
{
    double s = 0;
    for(double x : v) s += x;
    return v.empty() ? std::nan("") : s / v.size();
}

} // namespace cdm

#endif
