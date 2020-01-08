#ifndef LENS_CPP
#define LENS_CPP

#include <array> // std::array
#include <opencv2/core/core.hpp>
#include "math.cpp"
#include "renderer.cpp"
#include "lens.h"

using cv::Mat;
using std::vector;

// ---- lensT class members: ----

// Default constructor
lensT::lensT(Mat &kappa_in, int x, int y) 
	: w(kappa_in.cols), h(kappa_in.rows)
{
	// Move lens and thereby set its origin
	move(x, y);

	// Store both CV_64F and CV_8U versions of kappa intended for further calc. and display
	if (kappa_in.depth() == CV_8U)
	{
		/**
		 * Create double from uchar such that grayscale value 255 translates to kappa=2 
		 * (arbitr. choice)
		 */
		kappa8u = kappa_in.clone();
		kappa_in.convertTo(kappa_in, CV_64F);
		normalize(kappa_in, kappa_in, 2, 0, cv::NORM_MINMAX); 
		kappa = kappa_in.clone();
	}
	else
	{
		/**
		 * Create uchar version from input type and apply logarithmic scaling, restricting 
		 * kappa to [10^(-2.5), 255/70], which is an arbitrary choice for the display intensity 
		 * limits of the image on the screen. Feel free to modify these choices.
		 */
		kappa = kappa_in.clone();
		Mat threes = Mat::ones(h, w, CV_64FC1) * 2.5;
		cv::log(kappa_in, kappa_in);
		kappa_in += threes;
		kappa_in *= 70; 
		kappa_in.convertTo(kappa8u, CV_8U);
	}

	// Compute lensing potential psi from kappa, then differentiate it to get the deflection field
	std::cout << "-> Performing Fourier transforms and convolution..." << std::endl;
	compute_psi_from_kappa();
	std::cout << "-> Creating deflection field and shear..." << std::endl;
	compute_derivatives_from_psi();
	update_cc_and_caustics(1);
}

// Move lens to a specific pixel position on the sky (requires w,h to be set)
void lensT::move(int x_pos, int y_pos)
{
	// Update origin and end points according to current w and h
	origin[0] = x_pos - w/2;
	origin[1] = y_pos - h/2;
	end_points[0] = origin[0] + w;
	end_points[1] = origin[1] + h;
}

// Get origin of area covered by lens in pixels
const int *lensT::get_origin() 
{
	return origin;
}

// Get width of lens area (px)
int lensT::get_width() 
{
	return w;
}

// Get height of lens area (px)
int lensT::get_height() 
{
	return h;
}

// Get lens convergence map
Mat &lensT::get_kappa() 
{
	return kappa;
}

// Get lens convergence map
Mat &lensT::get_kappa8u()
{
	return kappa8u;
}

// Get critical curve map
Mat &lensT::get_cc() 
{
	return cc_map;
}

// Get caustic map
Mat &lensT::get_caustics()
{
	return caustic_map;
}

// Check if pixel (x,y) lies within the region covered by lens pixel data
bool lensT::contains(int x, int y)
{
	return origin[0]< x and x < end_points[0] and origin[1] < y and y < end_points[1];
}

// Solve lens equation for given pixel, return source plane position
void lensT::raytrace_pixel(int x1, int x2, int rel1_safe, int rel2_safe, double scale_fac, double &y1, double &y2)
{
	// Solve the lens equation under the given constraints
	y1 = x1 - alpha1.at<double>(rel2_safe, rel1_safe) * scale_fac * weight;
	y2 = x2 - alpha2.at<double>(rel2_safe, rel1_safe) * scale_fac * weight;
}

// Compute lensing potential via convolution in Fourier space (this requires kappa to be initialized)
void lensT::compute_psi_from_kappa()
{
	/**
	 * Prepare discrete fast Fourier transforms (DFTs). We enlarge the input Mat by two 
	 * via zero-padding since the DFT kernel will have twice the size of the data to fit it 
	 * entirely. We then obtain the optimal DFT size to maximize the performance 
	 * (see OpenCV doc for details)
	 */
	int orig_w = kappa.cols;
	int orig_h = kappa.rows;
	int opt_2w = cv::getOptimalDFTSize(2*orig_w);
	int opt_2h = cv::getOptimalDFTSize(2*orig_h);
	cv::copyMakeBorder(kappa, kappa, 0, opt_2h-orig_h, 0, opt_2w-orig_w, cv::BORDER_CONSTANT);
	
	// Create the Green's function kernel G
	Mat G = Mat::zeros(opt_2h, opt_2w, CV_64FC1); 
	Mat G_hat, kappa_hat, product;
	fill_green_fct(G);

	// Apply DFT to G and kappa, multiply and backward transform their results to obtain psi
	cv::dft(G, G_hat, cv::DFT_REAL_OUTPUT);
	cv::dft(kappa, kappa_hat, cv::DFT_REAL_OUTPUT);
	cv::mulSpectrums(kappa_hat, G_hat, product, 0, false); 
	cv::idft(product, psi, cv::DFT_SCALE);

	// Crop all maps back to the original size of kappa
	cv::Rect crop_region(0, 0, orig_w, orig_h);
	psi = psi(crop_region);
	kappa = kappa(crop_region);
}

// Compute alpha and shear by applying derivatives to psi (this requires psi to have been defined earlier)
void lensT::compute_derivatives_from_psi()
{
	// Derive deflection angle
	deriv_x(psi, alpha1);
	deriv_y(psi, alpha2);

	// Compute second derivatives
	Mat psi_11, psi_22, psi_12;
	deriv_x(alpha1, psi_11);
	deriv_y(alpha2, psi_22);
	deriv_y(alpha1, psi_12);

	// Compute shear from combination of these derivatives
	Mat diff = psi_11 - psi_22;
	shear = 0.25 * diff.mul(diff) + psi_12.mul(psi_12);
	cv::sqrt(shear, shear);
}


// (Re)-compute critical lines and caustics via the Jacobian from pre-computed kappa and shear 
void lensT::update_cc_and_caustics(bool include_radial_lines)
{
	/** 
	 * Compute derivatives if this wasn't done before (need to do this
	 * only once, then re-scale the quantities by the currently applied weight)
	 */
	if (shear.cols == 0)
		compute_derivatives_from_psi();

	// Define auxiliary map representing the unit matrix
	int width = kappa.cols;
	int height = kappa.rows;
	Mat unity = Mat::ones(height, width, CV_64FC1);

	/**
	 * Compute eigenvalues of the Jacobian matrix and obtain Jacobian determinant 
	 * (or tangential eigenvalue) map "detJ" from the eigenvalues.
	 */
	Mat tan_eigenval = unity - weight * (kappa + shear);
	Mat rad_eigenval = unity - weight * (kappa - shear);
	Mat detJ = (include_radial_lines) ? tan_eigenval.mul(rad_eigenval) : tan_eigenval;

	// Apply smoothing kernel to remove numerical pixel artifacts in the contours
	cv::GaussianBlur(detJ, detJ, cv::Size(0,0), 4);

	// Initialize cc_map if it wasn't done yet, then fill with binary data of regions with detJ < 0
	if (cc_map.cols != width and cc_map.rows != height)
		cc_map = Mat::zeros(height, width, CV_8UC1);
	cv::parallel_for_(cv::Range(0, height), binary_img_from_sign(detJ, cc_map));

	// Auxiliary quantities for contour drawing
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	int mode = cv::RETR_LIST;
	int method = cv::CHAIN_APPROX_SIMPLE;
	cv::Scalar white(255);
	int thickness = 1;
	int linestyle = 16; // Anti-aliased mode

	// Apply contour recognition and replace binary map with the contours
	findContours(cc_map, contours, hierarchy, mode, method);
	cc_map = Mat::zeros(height, width, CV_8UC1);
	for (unsigned i = 0; i < contours.size(); ++i)
		drawContours(cc_map, contours, i, white, thickness, linestyle);

	// As a second step, derive also the caustic lines by inversion of the CC map
	caustic_map = Mat::zeros(h, w, CV_8UC1);	
	cv::parallel_for_(cv::Range(0, h), invert_cc_map(this));

}


// ---- sourceT class members: ----

// Create source object
sourceT::sourceT(Mat &imageRGB_, int x_pos, int y_pos) : imageRGB(imageRGB_)
{
	// Split RGB img into single channels, keep copy of original img in imageRGB, get w and h
	split(imageRGB, channels);
	w = imageRGB.cols;
	h = imageRGB.rows;

	// Place source center at given pos. This will define its origin
	move(x_pos, y_pos);
}

// Move source center to a specific pixel position on the screen
void sourceT::move (int x_pos, int y_pos)
{
	pos[0] = x_pos;
	pos[1] = y_pos;
	origin[0] = x_pos - w/2;
	origin[1] = y_pos - h/2;
	end_points[0] = origin[0] + w;
	end_points[1] = origin[1] + h;
}

// Get width of the original (non-lensed) source image in px
int sourceT::get_width() 
{
	return w;
}

// Get height of the original (non-lensed) source image in px
int sourceT::get_height() 
{
	return h;
}

// Get origin of the original area covered by the source on the sky
const int *sourceT::get_origin() 
{
	return origin;
}

// Get current position of the source center
const int *sourceT::get_pos() 
{
	return pos;
}

// Get source image as R,G,B channels (array of Mat objects)
Mat (&sourceT::get_img())[3] 
{
	return channels;
}

// Resize source (keep orig. size for "imageRGB"; store current scaled version in "channels")
void sourceT::resize_area (double factor)
{
	// Remember current position of source center and size
	int orig_xpos = origin[0] + w/2;
	int orig_ypos = origin[1] + h/2;
	w = imageRGB.cols * factor;
	h = imageRGB.rows * factor;

	// Resize the image
	Mat rescaled;
	if (w > 0 and h > 0)
	{
		cv::resize(imageRGB, rescaled, cv::Size(), factor, factor);
		cv::split(rescaled, channels);
	}

	// Update source origin to keep lens centered on previous position
	move(orig_xpos, orig_ypos);
}

// Check if coordinate lies within area covered by source pixel data
bool sourceT::contains(double x_, double y_)
{
	int x = static_cast<int>(x_);
	int y = static_cast<int>(y_+0.5);
	return origin[0] < x and x < end_points[0] and origin[1] < y and y < end_points[1];
}

/**
 * Return source pixel at the given coordinate, using linear interpolation between neighboring 
 * pixels to obtain the contributions at a particular coordinate (which in general will lie 
 * between different pixels).
 */
cv::Vec3b sourceT::get_linear_interpolated_pixel(double beta1, double beta2)
{
	// If beta is outside the area covered by source img, display zero...
	cv::Vec3b val_to_show(0, 0, 0);
	if (!contains(beta1, beta2))
		return val_to_show;

	// Abbreviations
	double rel_beta1 = beta1 - origin[0];
	double rel_beta2 = beta2 - origin[1];
	double fl1 = floor(rel_beta1);
	double fl2 = floor(rel_beta2);
	unsigned low1 = relocate(fl1, w);
	unsigned low2 = relocate(fl2, h);
	unsigned up1 = relocate(fl1+1., w);
	unsigned up2 = relocate(fl2+1., h);
	double x = rel_beta1 - fl1;
	double y = rel_beta2 - fl2;
	double xy = x*y;

	// Define coefficient matrix for linear interpolation
	double c00 = 1.-x-y+xy;
	double c01 = x-xy;
	double c10 = y-xy;
	double c11 = xy;

	// Perform linear interpolated raytracing for each channel R,G,B
	for (size_t c = 0; c < 3; ++c)
	{
		unsigned I00 = channels[c].at<uchar>(low2, low1);
		unsigned I01 = channels[c].at<uchar>(low2, up1);
		unsigned I10 = channels[c].at<uchar>(up2, low1);
		unsigned I11 = channels[c].at<uchar>(up2, up1);
		val_to_show[c] = I00*c00 + I01*c01 + I10*c10 + I11*c11;
	}

	return val_to_show;
}


#endif
