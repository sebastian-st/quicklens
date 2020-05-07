#ifndef MATH_H
#define MATH_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using cv::Mat;

/**
 * Relocate pixel coordinate target: Replace float by nearest int within [0, interval_len-1]
 *
 * @param phys_coord 1D coordinate to relocate
 * @param interval_len Length of interval
 * @return The relocated / "safe" pixel value as integer
 */
int relocate(double phys_coord, int interval_len);

/**
 * Relocate pixel coordinate target: Replace int by nearest int within [0, interval_len-1]
 * 
 * @param rounded_coord 1D coordinate to relocate
 * @param interval_len Length of interval
 * @return The relocated / "safe" pixel value
 */
int relocate(int rounded_coord, int interval_len);

/**
 * Specify exponential fall-off of alpha assumed outside the area in which we have pixel data for the lens
 *
 * @param rel_px Relative coordinate with respect to lens origin
 * @param len Total interval length
 * @param half Half interval length
 * @param lm1 Length minus 1px
 * @return The fall-off weight "f_{i/j}" in 1D for the given pixel location (for 2D, multiply f_i*f_j)
 */
double exp_fall_off(int rel_px, int len, double half, double lm1);

/**
 * Relocate pixel coordinate target *and* specify exponential fall-off of alpha assumed outside the area 
 * in which we have lens pixel data
 *
 * @param[in] rel_px Relative coordinate with respect to lens origin
 * @param[in] len Total interval length
 * @param[in] half Half interval length
 * @param[in] lm1 Legth minus 1px
 * @param[out] safe_rel_x Relocated pixel value
 * @return weight The fall-off weight "f_{i/j}" in 1D for the given pixel (for 2D, multiply f_i*f_j)
 */
double relocate_and_compute_exp_falloff(int rel_px, int len, double half, double lm1, int &safe_rel_x);

/**
 * Fill the Green's function kernel
 * @param[out] green_fct Kernel to fill (needs to be initialized with zero and to have the desired width and height)
 */
void fill_green_fct(Mat &green_fct);

// Define enum for direction of image translations
enum Direction{
    ShiftUp=1, ShiftRight, ShiftDown, ShiftLeft
   };

/**
 * Shift all pixel coordinates by N pixels into given direction (required for finite differentiation)
 *
 * @param f Matrix to translate
 * @param N Number of pixels to shift image by
 * @param direction Direction to shift image into
 * @return The translated Mat image
 */
Mat translateImg(Mat f, int N, Direction direction);

/**
 * Compute partial numerical derivative of scalar field in x-direction
 *
 * @param[in] input Input Matrix (needs to be CV_64FC1)
 * @param[out] result Result Matrix (returns CV_64FC1)
 */
void deriv_x(Mat &input, Mat &result);

/**
 * Compute partial numerical derivative of scalar field in y-direction
 * 
 * @param[in] input Input Matrix (needs to be CV_64FC1)
 * @param[out] result Result Matrix (returns CV_64FC1)
 */
void deriv_y(Mat &input, Mat &result);

/**
 * Compute median of a Mat image
 * @param img_orig Input matrix (CV_64FC1)
 * @return Median
 */
double calculate_median(Mat img_orig);


#endif
