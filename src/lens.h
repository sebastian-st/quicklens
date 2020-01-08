#ifndef LENS_H
#define LENS_H

#include <opencv2/core/core.hpp>

using cv::Mat;

/**
 * @brief Class implementing a gravitational lens, its physical properties and its screen geometry.
 */
class lensT
{
	private:
		// Position and geometry
		int origin[2] = {0, 0};
		int end_points[2] = {0, 0};
		const int w;
		const int h;

		// Meshgrids
		Mat psi;	// Lensing potential
		Mat alpha1;	// Deflection angle field component in x direction
		Mat alpha2;	// Deflection angle field component in y direction
		Mat kappa, kappa8u; 	// Convergence
		Mat shear;	// Shear magnitude
		Mat cc_map;	// Critical curve contour map
		Mat caustic_map;	// Caustic map

		friend class invert_cc_map;

	public:
		// User defined weight factor to re-scale convergence
		double weight = 1.;

		/** 
		 * Constructor
		 *
		 * @param kappa_in Input convergence map (CV_64FC1)
		 * @param x Lens center x-position
		 * @param y Lens center y-position
		 */
		lensT(Mat &kappa_in, int x, int y);

		/**
		 * Move lens to a specific pixel position, update origin (requires w,h to be set!)
		 *
		 * @param x_pos Lens center (new) x-position
		 * @param y_pos Lens center (new) y-position
		 */
		void move(int x_pos, int y_pos);	

		/**
		 * Get origin of area covered by lens in pixels
		 * @return Current origin (x,y) coordinates
		 */
		const int *get_origin();
		
		/**
		 * Get width of lens area (px)
		 * @return Width
		 */
		int get_width();	

		/**
		 * Get height of lens area (px)
		 * @return Height
		 */
		int get_height();	

		/**
		 * Get lens convergence map
		 * @return Convergence map in CV_64FC1 (double) format
		 */
		Mat &get_kappa();

		/**
		 * Get lens convergence map
		 * @return Convergence map in CV_8UC1 (uchar) format
		 */
		Mat &get_kappa8u();
		
		/**
		 * Get critical curve map
		 * @return Critical curve contour map in CV_8UC1 (uchar) format
		 */
		Mat &get_cc();

		/**
		 * Get caustic map
		 * @return Caustic contour map in CV_8UC1 (uchar) format
		 */
		Mat &get_caustics();
		
		/**
		 * Check if pixel (x,y) lies within the region covered by lens pixel data
		 *
		 * @param x Pixel x-coordinate
		 * @param y Pixel y-coordinate
		 * @return Whether the pixel (x,y) is contained in the area
		 */
		bool contains(int x, int y);

		/**
		 * Solve lens equation for given pixel, return source plane position y. 
		 * @details Outside the lens area, alpha is interpolated exponentially to zero. 
		 *
		 * @param[in] x1 Lens plane pixel x-coordinate
		 * @param[in] x2 Lens plane pixel y-coordinate
		 * @param[in] rel1_safe Lens plane x coord rel to lens origin (has to be within range; passed here in addition to x1 to enable faster computation)
		 * @param[in] rel2_safe Lens plane y coord rel to lens origin (has to be within range; passed here in addition to x2 to enable faster computation)
		 * @param[in] scale_fac Add. scaling factor used to implement interpolation beyond bounds
		 * @param[out] y1 Target source plane x-coordinate
		 * @param[out] y2 Target source plane y-coordinate
		 */
		void raytrace_pixel(int x1, int x2, int rel1_safe, int rel2_safe, double scale_fac, double &y1, double &y2);

		/**
		 * Compute lensing potential psi from convergence via superposition with Green's fct
		 * (this requires that kappa has been defined)
		 */
		void compute_psi_from_kappa();

		/**
		 * Compute deflection field and shear magnitude by applying derivatives to psi
		 * (This requires psi to have been defined earlier)
		 */
		void compute_derivatives_from_psi();

		/**
		 * (Re)-compute critical lines and caustics via the Jacobian from pre-computed kappa and shear
		 */
		void update_cc_and_caustics(bool include_radial_lines);
};

/**
 * @brief Class representing a source and its geometric properties on the screen.
 */
class sourceT
{
	private:
		// Position and geometry
		int origin[2] = {0, 0};
		int pos[2] = {0, 0};
		int end_points[2] = {0, 0};
		int w, h;

		// Meshgrids
		Mat imageRGB, channels[3];
	public:
		/**
		 * Constructor
		 *
		 * @param imageRGB_ Input RGB composite image of the source to be lensed
		 * @param x_pos Lens center x pixel coordinate
		 * @param y_pos Lens center y pixel coordinate
		 */
		sourceT(Mat &imageRGB_, int x_pos, int y_pos);

		/**
		 * Move source center + update origin according to size (requires w,h to be set)
		 *
		 * @param x_pos Lens center (new) x pixel coordinate
		 * @param y_pos Lens center (new) y pixel coordinate
		 */
		void move (int x_pos, int y_pos);

		/**
		 * Get current width of the non-lensed source image in px
		 * @return Current width
		 */
		int get_width();

		/**
		 * Get current height of the non-lensed source image in px
		 * @return Current height
		 */
		int get_height();

		/**
		 * Get origin of the area covered by the (non-lensed) source on the sky
		 * @return Origin (x,y) coordinates
		 */
		const int *get_origin();

		/**
		 * Get current position of the source center
		 * @return Source center (x,y) coordinates
		 */
		const int *get_pos();

		/**
		 * Return reference to the source image channels 
		 * @return Array of Mat objects (R,G,B)
		 */
		Mat (&get_img())[3];

		/**
		 * Resize source (keep original "imageRGB"; store re-scaled version in "image")
		 * @param factor Factor by which the angular size of the source is resized
		 */
		void resize_area (double factor);

		/**
		 * Check if coordinate lies within area covered by source pixel data
		 *
		 * @param x_ X-coordinate
		 * @param y_ Y-coordinate
		 */
		bool contains(double x_, double y_);

		/**
		 * Get pixel at given coordinate, applying linear interpolation between neighboring pixels
		 *
		 * @param beta1 Input x coordinate
		 * @param beta2 Input y coordinate
		 * @return The value at (beta1, beta2) computed from the nearest neighbors (RGB value)
		 */
		cv::Vec3b get_linear_interpolated_pixel(double beta1, double beta2);
};

#endif
