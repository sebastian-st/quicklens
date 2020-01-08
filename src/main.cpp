/**
 *
 * quicklens - a fast gravitational lensing visualization tool
 * (c) 2019 by Sebastian Stapelberg
 * 
 * main.cpp: Initialization and main loop
 *
 **/

// Standard includes
#include <iostream>	// std::cout()
#include <math.h>	// std::floor()
#include <algorithm>	// std::min()
#include <string>

// OpenCV (Fast image manipulation / matrix calculations + very basic GUI features)
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Project includes
#include "math.cpp" 	// Auxiliary functions
#include "lens.cpp" 	// Physical objects
#include "screen_io.cpp"// Screen rendering, file I/O

/** 
 * Main function: Load image and perform all the necessary calculations that can be done beforehand 
 * (e.g. compute lensing potential from convergence via convolution)
 */
int main(int argc, char** argv)
{
	using std::cout;
	using std::endl;

	// Display program name, check number of cmd line arguments
	cout << "quicklens v1" << endl;
	if (argc < 3)
	{
		cout << "Usage: %prog lensfile sourcefile [N_threads (default:all)]" << endl;
		return -1;
	}

	// Get number of threads from argument list (default: use all threads)
	if (argc == 4)
	{
		char *eptr;
		int N_threads = std::strtol(argv[3], &eptr, 0);
		cv::setNumThreads(N_threads);
	}
	cout << "Started with " << cv::getNumThreads() << " threads" << endl;

	// Get filename for lens convergence and source image
	std::string lens_fn = argv[1];
	std::string fn = argv[2];

	// Display settings (feel free to adapt this to your needs)
	int resize_w = 1024;
	int resize_h = 768;

	/**
	 * Load source image (*.PNG, *.JPG, ...) as RGB color image (note: OpenCV uses "BGR" ordering).
	 * This yields a Mat object of type CV_8UC3 (3-channel uchar).
	 */
	cv::Mat imageRGB = cv::imread(fn, cv::IMREAD_COLOR);
	if (!imageRGB.data)
	{
		cout << "Error opening image file..." << endl;
		return -1;
	}
	
	/**
	 * Load lens convergence distribution (*.FITS, *.PNG, *.JPG, ...) as grayscale,
	 * creating CV_64FC1 (1-channel double) or CV_8UC1 (1-channel uchar) depending on the 
	 * image format - which we then convert into CV_64FC1 with a suitable normalization,
	 * in order to have an array of floating point numbers.
	 */
	cv::Mat kappa_input;
	int has_fits_input = false;
	#if HAS_CCFITS == TRUE
	try
	{
		// Try to read as FITS file. If this doesn't succeed, ...
		readmap(lens_fn, kappa_input);
		has_fits_input = true;
	}
	catch (CCfits::FitsException&)
	{
		has_fits_input = false;
	}
	#endif
	if (!has_fits_input) {
		// ...read as normal PNG, JPG or any other file format supported by OpenCV
		kappa_input = cv::imread(lens_fn, cv::IMREAD_GRAYSCALE);
		if (!kappa_input.data)
		{
			cout << "Error opening the image file..." << endl;
			return 0;
		}
	}

	// Create lens, source and screen objects (the latter opens an OpenCV window)
	int max_w = std::min(kappa_input.cols, imageRGB.cols);
	int max_h = std::min(kappa_input.rows, imageRGB.rows);
	const char* win = "CV_Window_";
	cout << "Creating lens, source and screen..." << endl;
	lensT lens(kappa_input, max_w/2, max_h/2);
	sourceT source(imageRGB, max_w/2, max_h/2);
	screenT screen(win, max_w, max_h, resize_w, resize_h, lens, source);

	// Enter refresh loop waiting for key/mouse event. The loop is exited with "q" or window close
	while (true)
	{
		if (cv::waitKey(200) == 113 or cv::getWindowProperty(win, cv::WND_PROP_AUTOSIZE) == -1)
			break;
		screen.clear_msg_display();
	}

	cv::destroyAllWindows();
	cout << "Closed." << endl;
}
