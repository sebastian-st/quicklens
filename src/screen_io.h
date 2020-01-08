#ifndef SCREEN_IO_H
#define SCREEN_IO_H

#include <chrono>
#include <opencv2/core/core.hpp>
#include "lens.h"

using cv::Mat;

/**
 * @brief Class representing interactive screen on which the images are rendered, using OpenCV windows
 */
class screenT
{
	private:
		// Window parameters
		int max_w = 0, max_h = 0;
		const char* win;

		// Objects to display
		lensT &lens;
		sourceT &src;
		Mat lensedRGB; // Lensed image
		Mat finalRGB; // Final image (lensed + overlays)

		// Drawing mode + trackbar params
		bool mouse_lbutton_down = false;
		int weight_int = 100;
		int source_size = 100;
		int overlay_mode = 1;

		// Internal settings and status variables
		bool redraw_cc_on_next_action = true;
		bool cc_radial = false;
		std::chrono::time_point<std::chrono::steady_clock> clock_start;
		std::string current_text = "";

		friend class Parallel_renderer;

	public:

		/**
		 * Constructor: Create window with screen and trackbars
		 *
		 * @param title Window name for OpenCV
		 * @param w Screen width
		 * @param h Screen height
		 * @param resize_w Resize window to a fix value independent of screen width
		 * @param resize_h Resize window to a fix value independent of screen height
		 * @param l Lens object to use for rendering
		 * @param s Source object to be displayed
		 */
		screenT(const char* title, int w, int h, int resize_w, int resize_h, lensT &l, sourceT &s);

		/**
		 * Compute and render lensed image
		 * @param redraw_overlay_only Re-draw only overlays? (I.e. re-use previous lensed image)
		 */
		void render_lensed_image(bool redraw_overlay_only = false);

		/**
		 * Refresh image on screen (e.g. re-render and show image)
		 * @param redraw_overlay_only Re-draw only overlays? (I.e. re-use previous lensed image)
		 */
		void refresh(bool redraw_overlay_only=false);

		/**
		 * Signal handler for mouse events
		 *
		 * @param sig Type of event (OpenCV)
		 * @param target_x X position of mouse click
		 * @param target_y Y position of mouse click
		 * @param std_scr Specific screen object
		 */
		static void handle_mouse_input(int sig, int target_x, int target_y, int, void *std_scr);

		/**
		 * Re-apply lens weight, recompute and re-draw the resulting critical curves.
		 * (Only requires applying a scaling factor to the pre-computed derivatives of psi)
		 * @param std_scr Specific screen object
		 */
		static void reapply_weight(int, void *std_scr);

		/**
		 * Refresh the overlays for critical curves, caustics or kappa. Don't re-render the lensed
		 * image.
		 * @param std_scr Specific screen object
		 */
		static void update_overlays(int, void *std_scr);

		/**
		 * Change angular size of source and update image on screen
		 * @param std_screen Specific screen object
		 */
		static void resize_source(int, void *std_screen);

		/**
		 * Clear the message display on the screen if sufficient time has passed.
		 */
		int clear_msg_display();
};

#if HAS_CCFITS == TRUE
/**
 * @brief Function for importing *.FITS image data into a Mat array
 * @param[in] filename Filename of FITS-image to import
 * @param[out] cv_image Mat object for storing FITS image (can be empty, returns type CV64FC1)
 **/
void readmap(std::string filename, Mat &cv_image);
#endif


#endif
