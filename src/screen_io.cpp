#ifndef SCREEN_IO_CPP
#define SCREEN_IO_CPP

#include <valarray>
#include <chrono>

// OpenCV core modules + high-level gui
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Libraries needed for handling the FITS file format
#if HAS_CCFITS == TRUE
#include <CCfits/CCfits>
#include <valarray>
#endif

// Custom resources
#include "lens.h"
#include "screen_io.h"
#include "renderer.cpp"

using cv::Mat;
using cv::Vec3b;
using namespace std::chrono;

// Create window with screen and trackbars
screenT::screenT(const char* title, int w, int h, int resize_w, int resize_h, lensT &l, sourceT &s) 
	: max_w(w), max_h(h), win(title), lens(l), src(s)
{
	// Initialize channels for lensed image and final image (i.e. lensed + overlays)
	lensedRGB = Mat::zeros(h, w, CV_8UC3);
	finalRGB = Mat::zeros(h, w, CV_8UC3);

	// Create OpenCV window with trackbars and mouse callback
	cv::namedWindow(win, cv::WINDOW_NORMAL);
	cv::resizeWindow(win, resize_w, resize_h);
	cv::createTrackbar("Overlays", win, &overlay_mode, 4, update_overlays, this);
	cv::createTrackbar("Kappa weight", win, &weight_int, 200, reapply_weight, this);
	cv::createTrackbar("Source size", win, &source_size, 400, resize_source, this);
	cv::setMouseCallback(win, handle_mouse_input, this);

	// Update the lens
	reapply_weight(0, this);
	clock_start = steady_clock::now();
}

// Compute and render the image; write the image data to result
void screenT::render_lensed_image(bool redraw_overlay_only)
{
	// Parallel computation/rendering of the image (defined in renderer.cpp)
	cv::parallel_for_(cv::Range(0, max_h), Parallel_renderer(this, !redraw_overlay_only));

	// Mark source center by a dot if wished
	bool mark_source = overlay_mode >= 2;
	int pos1 = src.get_pos()[0];
	int pos2 = src.get_pos()[1];
	if (mark_source)
		cv::circle(finalRGB, {pos1, pos2}, 7, cv::Scalar::all(210), -1);
}


// Refresh screen
void screenT::refresh(bool redraw_overlay_only)
{
	// Compute image, merge channels and mark the source position by a dot if wished
	render_lensed_image(redraw_overlay_only);

	if (current_text == "")
		cv::imshow(win, finalRGB);
	else
	{
		Mat tmpRGB = finalRGB.clone();
		int linestyle = 16;
		int sum = max_w + max_h;
		double sum_red = sum/(1920.+1080.);
		int text_pos1 = static_cast<int>(0.01*sum);
		int text_pos2 = static_cast<int>(0.02*sum);
		cv::Point pos(text_pos1, text_pos2);
		int font = cv::FONT_HERSHEY_SIMPLEX;
		cv::putText(tmpRGB, current_text, pos, font, sum_red, cv::Scalar::all(255), 2, linestyle);
		cv::imshow(win, tmpRGB);

	}

}

// Handle incoming mouse events (e.g. move source)
void screenT::handle_mouse_input(int sig, int target_x, int target_y, int, void *std_scr)
{
	screenT *scr = static_cast<screenT*>(std_scr);
	if (scr->mouse_lbutton_down and sig == cv::EVENT_MOUSEMOVE)
	{
		scr->lens.move(target_x, target_y);
		scr->refresh();
	}
	else if (sig == cv::EVENT_LBUTTONUP)
	{
		scr->mouse_lbutton_down = false;
	}
	else if (sig == cv::EVENT_LBUTTONDOWN)
	{
		scr->mouse_lbutton_down = true;
		scr->lens.move(target_x, target_y);
		scr->refresh();
	}
}

/**
 * Re-apply weight to the lens displayed on this screen, recompute and re-draw the resulting 
 * critical curves. (Only requires a scaling factor to be applied to pre-computed derivatives of psi)
 */
void screenT::reapply_weight(int, void *std_scr)
{
	screenT *scr = static_cast<screenT*>(std_scr);
	scr->lens.weight = static_cast<double>(scr->weight_int) / 20.;
	bool show_cc = (scr->overlay_mode > 1 and scr->overlay_mode <= 4);
	bool show_radial = (scr->overlay_mode == 3 or scr->overlay_mode == 4);
	if (show_cc)
	{
		scr->lens.update_cc_and_caustics(show_radial);
		scr->redraw_cc_on_next_action = false;	
	}
	else
		scr->redraw_cc_on_next_action = true;
	scr->refresh();
}

// Refresh the overlays for critical curves, caustics or kappa. Don't re-render the lensed image.
void screenT::update_overlays(int, void *std_scr)
{
	screenT *scr = static_cast<screenT*>(std_scr);
	bool show_cc = (scr->overlay_mode > 1 and scr->overlay_mode <= 4);
	if (show_cc)
	{
		bool show_radial = (scr->overlay_mode == 3 or scr->overlay_mode == 4);
		if (scr->cc_radial != show_radial)
		{
			scr->redraw_cc_on_next_action = true;
			scr->cc_radial = show_radial;
		}

		if (scr->redraw_cc_on_next_action)
		{
			scr->lens.update_cc_and_caustics(show_radial);
			scr->redraw_cc_on_next_action = false;
		}
	}

	switch (scr->overlay_mode)
	{
		case 1 : scr->current_text = "Add lens convergence"; break;
		case 2 : scr->current_text = "Add critical curves (t) + source center (dot)"; break;
		case 3 : scr->current_text = "Add critical curves (t+r) + source center (dot)"; break;
		case 4 : scr->current_text = "Add lens + critical curves + source center (dot)"; break;
		default : scr->current_text = "";
	}
	scr->refresh(1);
	scr->clock_start = steady_clock::now();
}

// Resize the angular extent of the source displayed on the screen and update image on screen
void screenT::resize_source(int, void *std_screen)
{
	screenT *screen = static_cast<screenT*>(std_screen);

	// Apply the changed source_size paramter controlled by the trackbar
	double factor = static_cast<double>(screen->source_size)/100.;
	screen->src.resize_area(factor);
	screen->refresh();
}

// Clear the message display on the screen if sufficient time has passed.
int screenT::clear_msg_display()
{
	if (current_text.empty())
		return 1;

	time_point<steady_clock> clock_end = steady_clock::now();
	duration<double> timespan = duration_cast<duration<double> >(clock_end - clock_start);
	
	if (timespan.count() < 1)
		return 1;

	current_text = "";
	cv::imshow(win, finalRGB);
	return 1;
}

#if HAS_CCFITS == TRUE
// Function for importing *.FITS image data into a Mat array
void readmap(string filename, Mat &cv_image)
{
	// Open FITS file and load primary HDU data into valarray
	CCfits::FITS infile(filename, CCfits::Read, true);
	CCfits::PHDU &img = infile.pHDU();
	std::valarray<double> contents;
	img.read(contents);
	size_t w(img.axis(0));
	size_t h(img.axis(1));

	// Read the contents into Mat image
	cv_image = Mat::zeros(h, w, CV_64FC1);
	for (size_t i = 0; i < h; i++)
		for(size_t j = 0; j < w; j++)
		{
			double val = contents[i*w + j];
			if (std::isnan(val))
				cv_image.at<double>(h-1-i, j) = 0.;
			else
				cv_image.at<double>(h-1-i, j) = val;

		}	
}
#endif

#endif
