#include <opencv2/core/core.hpp>

#include "math.h"
#include "lens.h"
#include "screen_io.h"
#include "renderer.h"

using cv::Mat;
using cv::Vec3b;


/**
 * Parallel_renderer Constructor
 * @param std_screen Screen object referred to
 * @param mode Drawing mode: Re-draw whole image? (rather than re-drawing only the overlays)
 */
Parallel_renderer::Parallel_renderer(screenT *std_screen, bool mode) : screen(std_screen), recompute_lensed(mode) {}

void Parallel_renderer::operator()(const cv::Range &range) const
{
	// Useful abbreviations
	lensT &lens = screen->lens;
	sourceT &src = screen->src;
	Mat &lensedRGB = screen->lensedRGB;
	Mat &finalRGB = screen->finalRGB;
	const int h = lens.get_height();
	const int w = lens.get_width();
	const double hd = static_cast<double>(h);
	const double wd = static_cast<double>(w);
	const double h2 = hd*0.5;
	const double w2 = wd*0.5;
	const double hm1 = hd-1.;
	const double wm1 = wd-1.;

	// Evaluate user-defined overlay mode parameters
	bool show_cc = (screen->overlay_mode > 1 and screen->overlay_mode <= 4);
	bool show_lens = (screen->overlay_mode == 1 or screen->overlay_mode == 4);
	bool show_overlays = (screen->overlay_mode > 0);

	// Parallel processing of loop over image pixels (j,i)
	for (int i = range.start; i < range.end; ++i)
	{
		/**
		 * Define a relative y-coordinate with respect to the lens origin and
		 * specify the fall-off of alpha outside the defined lens area in y-dir.
		 */
		int rel_i = i - lens.get_origin()[1];
		int safe_i = 0; // Use dummy values until "recompute_lensed" is checked
		double fi = 1.;
		if (recompute_lensed)
			fi = relocate_and_compute_exp_falloff(rel_i, h, h2, hm1, safe_i);

		for (int j = 0; j < screen->max_w; ++j)
		{
			// Define relative x-coordinate as well
			int rel_j = j - lens.get_origin()[0];

			// First consider the overlays (if these sum to 255, can skip raytracing)
			unsigned overlay_sum = 0;
			bool is_caustic_pixel = false;
			if (show_overlays and lens.contains(j, i))
			{
				// Check if pixel coincides with a CC of the lens
				unsigned cc_px_value = show_cc ? lens.get_cc().at<uchar>(rel_i, rel_j) : 0;
				if (show_cc and cc_px_value > 0)
				{	
					/**
					 * If CC val is 255 and if only the overlays shall be
					 * redrawn, we can set pixel directly to 255 
					 * and skip to next one, since CC is always on top.
					 */
					if (cc_px_value == 255 and !recompute_lensed)
					{
						finalRGB.at<Vec3b>(i,j) = Vec3b(255,255,255);
						continue;
					}

					/**
					 * Add contribution of the CC contour image. Gray 
					 * values can occur since contours are anti-aliased.
					 */	
					overlay_sum += cc_px_value;
				}

				// Check if pixel lies on a caustic (comes second after CC)
				if (lens.get_caustics().at<uchar>(rel_i, rel_j) > 0)
					is_caustic_pixel = show_cc;

				if (is_caustic_pixel)
				{
					finalRGB.at<Vec3b>(i,j) = Vec3b(0,0,255);
					if (!recompute_lensed)
						continue;
				}

				// Add lens convergence to overlays
				if (show_lens)
				{
					unsigned kappa8u = lens.get_kappa8u().at<uchar>(rel_i, rel_j);
					overlay_sum += kappa8u;
				}
			}

			// Perform the raytracing to compute the lensed image in the background
			if (recompute_lensed)
			{
				/**
				 * Place rel_j within bounds (result -> safe_j) and specify 
				 * the fall-off of alpha outside the defined lens area in 
				 * y-direction
				 */
				int safe_j;
				double fj = relocate_and_compute_exp_falloff(rel_j, w, w2, wm1, safe_j);
				double beta1, beta2;
				
				/**
				 * Compute lens eq. at pixel (j,i) to get target source pos.
				 * Then, get linearly interpolated source RGB value at target 
				 * (beta1, beta2). Since the latter will in general lie
				 * between different source pixels, the interpolation is used
				 * to obtain the contributions at the particular coord.
				 * The function returns zero if beta is outside the area 
				 * covered  by the source.
				 */
				lens.raytrace_pixel(j, i, safe_j, safe_i, fi*fj, beta1, beta2);
				lensedRGB.at<Vec3b>(i,j) = src.get_linear_interpolated_pixel(beta1, beta2);
			}

			/**
			 * Get final image pixel as the sum of lensed image + overlay (unless the 
			 * pixel belongs to a caustic, in which case it has already been set)
			 */
			if (!is_caustic_pixel)
				for (size_t c = 0; c < 3; ++c)
				{
					unsigned final_val = lensedRGB.at<Vec3b>(i,j)[c] + overlay_sum;
					if (final_val > 255)
						finalRGB.at<Vec3b>(i,j)[c] = 255;
					else
						finalRGB.at<Vec3b>(i,j)[c] = final_val;
				}
		
		}
	}
}


/**
 * Binary_img_from_sign class constructor
 * @param[in] input Input Mat image (needs to be of type CV_64F, i.e. double)
 * @param[out] output Output Mat image (returns CV_8U, i.e. uchar)
 */
binary_img_from_sign::binary_img_from_sign(Mat &input, Mat &output) : in(input), out(output) {}

void binary_img_from_sign::operator()(const cv::Range &range) const
{
	int width = in.cols;

	for (int c = range.start; c < range.end; ++c)
		for (int d = 0; d < width; ++d)
			out.at<uchar>(c,d) = in.at<double>(c,d) <= 0;
}


/**
 * Invert_cc_map parallelisation class constructor
 * @param[in] lens_ Lens whose cc_map and caustic_map we are referring to
 */
invert_cc_map::invert_cc_map(lensT *lens_) : lens(lens_) {}

void invert_cc_map::operator()(const cv::Range &range) const
{
	int width = lens->caustic_map.cols;
	int height = lens->caustic_map.rows;

	for (int i = range.start; i < range.end; ++i)
		for (int j = 0; j < width; ++j)
		{
			// Perform raytracing to map from lens plane to source plane.
			double beta1, beta2;
			lens->raytrace_pixel(j, i, j, i, 1., beta1, beta2);
			int b1 = static_cast<int>(beta1+0.5);
			int b2 = static_cast<int>(beta2+0.5);

			if (lens->cc_map.at<uchar>(i, j) > 0 and 0 <= b1 and b1 < width and 0 <= b2 and b2 < height)
			{
				lens->caustic_map.at<uchar>(b2, b1) = 255;

				// Compromise between interpolation, runtime and a reasonable line thickness: perform dilation with a "half" kernel (0,1,0) (1,1,0) (0,0,0)
				int low[2] = {relocate(b1-1, width), relocate(b2-1, height)};
				lens->caustic_map.at<uchar>(low[1], b1) = 255;
				lens->caustic_map.at<uchar>(b2, low[0]) = 255;
			}

		}
}


