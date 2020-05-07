#ifndef RENDERER_H
#define RENDERER_H

#include <opencv2/core/core.hpp>
#include "math.h"
#include "lens.h"
#include "screen_io.h"

using cv::Mat;
using cv::Vec3b;

/**
 * @brief Class for OpenCV parallelization: Render the image produced by the lens on the given screen for the given source
 */
class Parallel_renderer : public cv::ParallelLoopBody
{
	private:
		screenT *screen;
		bool recompute_lensed;
	public:
		/**
		 * Constructor
		 * @param std_screen Screen object referred to
		 * @param mode Drawing mode: Re-draw whole image? (rather than re-drawing only the overlays)
		 */
		Parallel_renderer(screenT *std_screen, bool mode);

		virtual void operator()(const cv::Range &range) const;
};


/**
 * @brief Class for OpenCV parallelization: Compute binary map by taking the sign a 1-channel image (< 0 yields "1")
 */
class binary_img_from_sign : public cv::ParallelLoopBody
{
	private:
		Mat &in;
		Mat &out;
	public:
		/**
		 * Constructor
		 * @param[in] input Input Mat image (needs to be of type CV_64F, i.e. double)
		 * @param[out] output Output Mat image (returns CV_8U, i.e. uchar)
		 */
		binary_img_from_sign(Mat &input, Mat &output);

		virtual void operator()(const cv::Range &range) const;
};

/**
 * @brief Class for OpenCV parallelization: Invert the critical curve map of a lens to derive its caustics
 */
class invert_cc_map : public cv::ParallelLoopBody
{       
	private:
		lensT *lens;
	public: 
		/**
		 * Constructor
		 * @param[in] lens_ Lens whose cc_map and caustic_map we are referring to
		 */
		invert_cc_map(lensT *lens_);
		
		virtual void operator()(const cv::Range &range) const;
};


#endif
