/* CDM apps: depth image I/O and visualisation. */

#ifndef CDM_IO_H
#define CDM_IO_H

#include <string>

#include <opencv2/core/core.hpp>

namespace cdm
{

/*! A uint16 depth PNG as CV_32FC1 metres (raw / depth_factor; raw 0 stays 0). Throws otherwise. */
cv::Mat LoadDepthMetres(const std::string &path, double depth_factor);

/*!
 * CV_32FC1 metres as CV_16UC1 at depth_factor units per metre, rounded. 0 is
 * "no depth". A depth beyond the uint16 range cannot be stored: it is written
 * as 0 and counted in `num_clipped`, never saturated to a false maximum.
 */
cv::Mat DepthToU16(const cv::Mat &depth_m, double depth_factor, int *num_clipped = nullptr);

/*!
 * Colour range for a sequence: 1.1 x the 95th percentile of the valid depth in
 * `depth_m`. Callers compute it once and keep it, so colours stay comparable
 * across frames.
 */
float AutoDepthRange(const cv::Mat &depth_m);

/*! Turbo colour map over [0, max_m] metres; missing (0) depth is black. */
cv::Mat ColorizeDepth(const cv::Mat &depth_m, float max_m);

/*! RGB | sensor depth | CDM depth, one colour scale, with a caption per panel. */
cv::Mat SideBySide(const cv::Mat &bgr, const cv::Mat &raw_m, const cv::Mat &cdm_m, float max_m,
                   const std::string &status);

} // namespace cdm

#endif
