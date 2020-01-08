#ifndef MATH_CPP
#define MATH_CPP

#include <iostream> // std::count
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
int relocate(double phys_coord, int interval_len)
{
	int nearest = static_cast<int>(phys_coord+0.5);
	if (nearest < 0)
		return 0;
	if (nearest >= interval_len)
		return interval_len-1;
	return nearest;
}

/**
 * Relocate pixel coordinate target: Replace int by nearest int within [0, interval_len-1]
 * 
 * @param rounded_coord 1D coordinate to relocate
 * @param interval_len Length of interval
 * @return The relocated / "safe" pixel value
 */
int relocate(int rounded_coord, int interval_len)
{
	if (rounded_coord < 0)
		return 0;
	if (rounded_coord >= interval_len)
		return interval_len-1;
	return rounded_coord;
}

/**
 * Specify exponential fall-off of alpha assumed outside the area in which we have pixel data for the lens
 *
 * @param rel_px Relative coordinate with respect to lens origin
 * @param len Total interval length
 * @param half Half interval length
 * @param lm1 Length minus 1px
 * @return The fall-off weight "f_{i/j}" in 1D for the given pixel location (for 2D, multiply f_i*f_j)
 */
double exp_fall_off(int rel_px, int len, double half, double lm1)
{
	double f = 1.;
	if (rel_px < 0)
		f *= exp(static_cast<double>(rel_px)/half);
	else if (rel_px >= len)
		f *= exp(static_cast<double>(lm1-rel_px)/half);
	return f;
}

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
double relocate_and_compute_exp_falloff(int rel_px, int len, double half, double lm1, int &safe_rel_x)
{
	if (rel_px < 0)
	{
		safe_rel_x = 0;
		return exp(static_cast<double>(rel_px)/half);
	}
	if (rel_px >= len)
	{
		safe_rel_x = lm1;
		return exp(static_cast<double>(lm1-rel_px)/half);
	}
	
	safe_rel_x = rel_px;
	return 1.;
}

/**
 * Fill the Green's function kernel
 * @param[out] green_fct Kernel to fill (needs to be initialized with zero and to have the desired width and height)
 */
void fill_green_fct(Mat &green_fct)
{
	size_t X = green_fct.cols;
	size_t Y = green_fct.rows;
	size_t X2 = X/2;
	size_t Y2 = Y/2;

	double factor = 1/M_PI;

	size_t i, j;
	for (i = 0; i <= Y2; ++i)
	{
		size_t Ymi = Y - i;
		int i_sq = i*i;

		for (j = 0; j <= X2; ++j)
		{
			size_t Xmj = X - j;
			double val;
			if (i == 0 and j == 0)
				val = -1.4658711977588554; // G(theta = 0.01) as a lower cut for the log
			else
				val = factor * log(sqrt(i_sq + j*j));

			// Compute values for one quarter and...
			green_fct.at<double>(i, j) = val;

			// ...exploit spherical symmetry to fill the remaining quarters
			if (i != 0)
			{
				green_fct.at<double>(Ymi, j) = val;
				if (j != 0)
					green_fct.at<double>(Ymi, Xmj) = val;
			}
			if (j != 0)
				green_fct.at<double>(i, Xmj) = val;
		}
	}
}


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
Mat translateImg(Mat f, int N, Direction direction)
{
    // Create a same sized temporary Mat with all the pixels flagged as invalid (-1)
    Mat tmp = Mat::zeros(f.size(), f.type());

    switch (direction)
    {
	    case(ShiftUp) :
		f(cv::Rect(0, N, f.cols, f.rows - N)).copyTo(tmp(cv::Rect(0, 0, tmp.cols, tmp.rows - N)));
		break;
	    case(ShiftLeft) :
		f(cv::Rect(0, 0, f.cols - N, f.rows)).copyTo(tmp(cv::Rect(N, 0, f.cols - N, f.rows)));
		break;
	    case(ShiftDown) :
		f(cv::Rect(0, 0, f.cols, f.rows - N)).copyTo(tmp(cv::Rect(0, N, f.cols, f.rows - N)));
		break;
	    case(ShiftRight) :
		f(cv::Rect(N, 0, f.cols - N, f.rows)).copyTo(tmp(cv::Rect(0, 0, f.cols - N, f.rows)));
		break;
	    default:
		std::cout << "Shift direction is not set properly" << std::endl;
	    }

    return tmp;
}

/**
 * Compute partial numerical derivative of scalar field in x-direction
 *
 * @param[in] input Input Matrix (needs to be CV_64FC1)
 * @param[out] result Result Matrix (returns CV_64FC1)
 */
void deriv_x(Mat &input, Mat &result)
{
	// Apply the finite differences
	Mat xp1, xm1;
	xp1 = translateImg(input, 1, ShiftRight);
	xm1 = translateImg(input, 1, ShiftLeft);
	result = (xp1-xm1)/2.0;

	int N_r = result.rows;
	int N_c = result.cols;
	for (int i = 0; i < N_r; ++i)
        {
		double border1 = result.at<double>(i, 2);
		double border2 = result.at<double>(i, N_c-3);
                result.at<double>(i, 0) = border1;
                result.at<double>(i, 1) = border1;
                result.at<double>(i, N_c-1) = border2;
                result.at<double>(i, N_c-2) = border2;
        }
}

/**
 * Compute partial numerical derivative of scalar field in y-direction
 * 
 * @param[in] input Input Matrix (needs to be CV_64FC1)
 * @param[out] result Result Matrix (returns CV_64FC1)
 */
void deriv_y(Mat &input, Mat &result)
{
	// Apply the finite differences
	Mat yp1, ym1;
	yp1 = translateImg(input, 1, ShiftUp);
	ym1 = translateImg(input, 1, ShiftDown);
	result = (yp1-ym1)/2.0;
	
	// Check the boundaries
	int N_c = result.cols;
	int N_r = result.rows;
	for (int i = 0; i < N_c; ++i)
        {
		double border1 = result.at<double>(2, i);
		double border2 = result.at<double>(N_r-3, i);
                result.at<double>(0, i) = border1;
                result.at<double>(1, i) = border1;
                result.at<double>(N_r-1, i) = border2;
                result.at<double>(N_r-2, i) = border2;
        }
}

/**
 * Compute median of a Mat image
 * @param img_orig Input matrix (CV_64FC1)
 * @return Median
 */
double calculate_median(Mat img_orig)
{
        // Create vector from Mat
	Mat img = img_orig.clone().reshape(0, 1);
	std::vector<double> flat;
	img.copyTo(flat);

	// Find median element
        nth_element(flat.begin(), flat.begin() + flat.size() / 2, flat.end());
        return flat[flat.size() / 2]; 
}

#endif
