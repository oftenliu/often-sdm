/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: examples/landmark_detection.cpp
 *
 * Copyright 2014, 2015 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "superviseddescent/superviseddescent.hpp"
#include "superviseddescent/regressors.hpp"

extern "C" {
	#include "hog.h" // From the VLFeat C library
}

#include "cereal/archives/binary.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <cassert>

using namespace superviseddescent;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using cv::Vec2f;
using std::vector;
using std::string;
using std::cout;
using std::endl;



cv::Rect boundingSquare(Mat pts, int n) {

   float minx = pts.at<float>(0), maxx = pts.at<float>(0);
   float miny = pts.at<float>(0), maxy = pts.at<float>(n);

   for (int i = 1; i < n; i++) {
       float x = pts.at<float>(i);
       float y = pts.at<float>(i + n);
       if (minx > x) minx = x;
       else if (maxx < x) maxx = x;

       if (miny > y) miny = y;
       else if (maxy < y) maxy = y;
   }

   cv::Rect bbox = cv::Rect(static_cast<int>(minx), static_cast<int>(miny),

   static_cast<int>(maxx-minx), static_cast<int>(maxy-miny));

   //bbox.width = cv::min(bbox.width, bbox.height);
   //bbox.height = bbox.width;
   return bbox;
}

cv::Rect extentRect(cv::Rect fr, float h, float w, float factor) {
   float eh = fr.height * factor, ew = fr.width * factor;
   float ex_1 = fr.x - (ew - fr.width)/ 2, ey_1 = fr.y - (eh - fr.height) / (1.25);
   float ex_2 = std::min(ex_1 + ew, w - 1), ey_2 =  std::min(ey_1 + eh, h - 1);

   cv::Rect res;
   res.x = std::max(int(ex_1), 0), res.y = std::max(int(ey_1), 0);
   res.height = ey_2 - res.y, res.width = ex_2 - res.x;

   return res;
}



/**
 * Reads an ibug .pts landmark file and returns an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @param[in] filename Path to a .pts file.
 * @return An ordered vector with the 68 ibug landmarks.
 */
std::pair<vector<Vec2f>, cv::Rect>   read_pts_landmarks(std::string filename) 
{
	using std::getline;
	vector<Vec2f> landmarks;
	landmarks.reserve(68);
	cv::Rect box;
	std::ifstream file(filename);
	if (!file.is_open()) {
		printf("Could not open landmark file: %s .\n" ,filename.c_str());
		return std::make_pair(landmarks, box); 
	}
	
	string line;
	// Skip the first 3 lines, they're header lines:
	getline(file, line); // 'version: 1'
	getline(file, line); // 'n_points : 68'
	getline(file, line); // '{'

	while (getline(file, line))
	{
		if (line == "}") { // end of the file
			break;
		}
		std::stringstream line_stream(line);
		Vec2f landmark(0.0f, 0.0f);
		if (!(line_stream >> landmark[0] >> landmark[1])) {
			throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
		}
		// From the iBug website:
		// "Please note that the re-annotated data for this challenge are saved in the Matlab convention of 1 being 
		// the first index, i.e. the coordinates of the top left pixel in an image are x=1, y=1."
		// ==> So we shift every point by 1:
		landmark[0] -= 1.0f;
		landmark[1] -= 1.0f;
		landmarks.emplace_back(landmark);
	}

	while (getline(file, line))
	{
		if (line != "{") { // end of the file
			continue;
		}
        break;
	}

	getline(file, line);
	box.x = atoi(line.c_str());	
	

	//printf("get line value is: %s and the final value is %d.\n" ,line.c_str(),box.x );

	getline(file, line);	
 	box.y = atoi(line.c_str());	
	//printf("get line value is: %s and the final value is %d.\n" ,line.c_str(),box.y );


	getline(file, line);	
	box.width = atoi(line.c_str());	
	//printf("get line value is: %s and the final value is %d.\n" ,line.c_str(),box.width );

	getline(file, line);	
	box.height = atoi(line.c_str());	
	//printf("get line value is: %s and the final value is %d.\n" ,line.c_str(),box.height );
    return std::make_pair(landmarks, box); 

};





// Loads images and their corresponding landmarks from .pts files into memory
/**
 * Loads all .png images and their corresponding .pts landmarks from the given
 * directory and returns them. For the sake of a simple example, the landmarks
 * are filtered and only 5 of the 68 landmarks are kept.
 *
 * @param[in] directory A directory with .png images and ibug .pts files.
 * @return A pair with the loaded images and their landmarks (all in one cv::Mat).
 */
std::pair<vector<std::string>, Mat> load_ibug_data(fs::path directory,vector<cv::Rect> &boxList)
{
	vector<std::string> images;
	Mat landmarks;

	// Get all the filenames in the given directory:
	// vector<fs::path> image_filenames;
	// fs::directory_iterator end_itr;
	// for (fs::directory_iterator i(directory); i != end_itr; ++i)
	// {
	// 	if (fs::is_regular_file(i->status()) && i->path().extension() == ".png")
	// 		image_filenames.emplace_back(i->path()); //类似于push_pack
	// }

    vector<fs::path> image_filenames;
    fs::recursive_directory_iterator end_iter;
    for(fs::recursive_directory_iterator iter(directory);iter!=end_iter;iter++){
        try{
            if (fs::is_directory( *iter ) ){
                std::cout<<*iter << "is dir" << std::endl;
                //ret.push_back(iter->path().string());
                //ScanAllFiles::scanFiles(iter->path().string(),ret);
            }else if(fs::is_regular_file(iter->status()) && (iter->path().extension() == ".png") || iter->path().extension() == ".jpg"){
                image_filenames.push_back(iter->path().string());
                std::cout << *iter << " is a file" << std::endl;
            }
        } catch ( const std::exception & ex ){
            std::cerr << ex.what() << std::endl;
            continue;
        }
	}


	// Load each image and its landmarks into memory:
	for (auto file : image_filenames)
	{
		images.emplace_back(file.string());
		// We load the landmarks and convert them into [x_0, ..., x_n, y_0, ..., y_n] format:
		file.replace_extension(".pts");

		cv::Rect box; 
		vector<Vec2f> lms;
		std::tie(lms, box) = read_pts_landmarks(file.string());

		int num_landmarks = 68;
		Mat landmarks_as_row(1, 2 * num_landmarks, CV_32FC1);

		// Store the landmarks 31, 37, 46, 49 and 55 for training:
		// (we subtract 1 because std::vector's indexing starts at 0, ibug starts at 1)
		for (int i = 0; i < num_landmarks;i++)
		{
			landmarks_as_row.at<float>(i) = lms[i][0];
			landmarks_as_row.at<float>(i + num_landmarks) = lms[i][1]; // y coordinate
		}
		// landmarks_as_row.at<float>(0) = lms[30][0]; // the x coordinate
		// landmarks_as_row.at<float>(0 + num_landmarks) = lms[30][1]; // y coordinate
		// landmarks_as_row.at<float>(1) = lms[36][0];
		// landmarks_as_row.at<float>(1 + num_landmarks) = lms[36][1];
		// landmarks_as_row.at<float>(2) = lms[45][0];
		// landmarks_as_row.at<float>(2 + num_landmarks) = lms[45][1];
		// landmarks_as_row.at<float>(3) = lms[48][0];
		// landmarks_as_row.at<float>(3 + num_landmarks) = lms[48][1];
		// landmarks_as_row.at<float>(4) = lms[54][0];
		// landmarks_as_row.at<float>(4 + num_landmarks) = lms[54][1];
	
		landmarks.push_back(landmarks_as_row);
		boxList.emplace_back(box);
	}
	return std::make_pair(images, landmarks);
};

/**
 * Function object that extracts HoG features at given 2D landmark locations
 * and returns them as a row vector.
 *
 * We wrap all the C-style memory allocations of the VLFeat library
 * in cv::Mat's.
 * Note: Any other library and features can of course be used.
 */
class HogTransform
{
public:
	/**
	 * Constructs a HoG transform with given images and parameters.
	 * The images are needed so the HoG features can be extracted from the images
	 * when the SupervisedDescentOptimiser calls this functor (the optimiser
	 * itself doesn't know and care about the images).
	 *
	 * Note: \p images can consist of only 1 image, when using the model for
	 * prediction on new images.
	 *
	 * Note: Only VlHogVariantUoctti is tested.
	 *
	 * @param[in] images A vector of images used for training or testing.
	 * @param[in] vlhog_variant The VLFeat HoG variant.
	 * @param[in] num_cells Number of HoG cells that should be constructed around each landmark.
	 * @param[in] cell_size Width of one HoG cell in pixels.
	 * @param[in] num_bins Number of orientations of a HoG cell. 方向划分个数
	 *///3 /*numCells*/, 12 /*cellSize*/, 4 /*numBins*/
	HogTransform(vector<std::string> imageFiles, int nType ,VlHogVariant vlhog_variant, int num_cells, int cell_size, int num_bins) :  imageFiles(imageFiles), nType(nType),vlhog_variant(vlhog_variant), num_cells(num_cells), cell_size(cell_size), num_bins(num_bins)
	{
	};


	HogTransform(Mat images, int nType ,VlHogVariant vlhog_variant, int num_cells, int cell_size, int num_bins) : images(images),nType(nType),  vlhog_variant(vlhog_variant), num_cells(num_cells), cell_size(cell_size), num_bins(num_bins)
	{
	};

	/**
	 * Uses the current parameters (the 2D landmark locations, in SDM
	 * terminology the \c x), and extracts HoG features at these positions.
	 * These HoG features are the new \c y values.
	 *
	 * The 2D landmark position estimates are given as a row vector
	 * [x_0, ..., x_n, y_0, ..., y_n].
	 *
	 * @param[in] parameters The current 2D landmark position estimate.
	 * @param[in] regressor_level Not used in this example.
	 * @param[in] training_index Gets supplied by the SupervisedDescent optimiser during training and testing, to know from which image to extract features.
	 * @return Returns the HoG features at the given 2D locations.
	 *///thread_pool.enqueue(projection, current_x.row(sample_index), regressor_level, sample_index)
	cv::Mat operator()(cv::Mat parameters, size_t regressor_level, int training_index = 0) //重载（）运算
	{
		assert(parameters.rows == 1);
		using cv::Mat;

		Mat image;
		if (nType == 1)
		{
			image = cv::imread(imageFiles[training_index]);	
		}
		else
		{
			image = images;
		}
		Mat gray_image;
		if (image.channels() == 3) {//转换为灰度图像
			cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
		}
		else {
			gray_image = image;
		}

		// Note: We could use the 'regressorLevel' to choose the window size (and
		// other parameters adaptively). We omit this for the sake of a short example.

		int patch_width_half = num_cells * (cell_size / 2);

		Mat hog_descriptors; // We'll get the dimensions later from vl_hog_get_*

		int num_landmarks = parameters.cols / 2;//关键点数量
		for (int i = 0; i < num_landmarks; ++i) {//遍历关键点
			int x = cvRound(parameters.at<float>(i)); //四舍五入
			int y = cvRound(parameters.at<float>(i + num_landmarks));

			Mat roi_img;
			if (x - patch_width_half < 0 || y - patch_width_half < 0 || x + patch_width_half >= gray_image.cols || y + patch_width_half >= gray_image.rows) {
				// The feature extraction location is too far near a border. We extend the
				// image (add a black canvas) and then extract from this larger image.
				int borderLeft = (x - patch_width_half) < 0 ? std::abs(x - patch_width_half) : 0; // x and y are patch-centers
				int borderTop = (y - patch_width_half) < 0 ? std::abs(y - patch_width_half) : 0;
				int borderRight = (x + patch_width_half) >= gray_image.cols ? std::abs(gray_image.cols - (x + patch_width_half)) : 0;
				int borderBottom = (y + patch_width_half) >= gray_image.rows ? std::abs(gray_image.rows - (y + patch_width_half)) : 0;
				Mat extendedImage = gray_image.clone();
				cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
				cv::Rect roi((x - patch_width_half) + borderLeft, (y - patch_width_half) + borderTop, patch_width_half * 2, patch_width_half * 2); // Rect: x y w h. x and y are top-left corner.
				roi_img = extendedImage(roi).clone(); // clone because we need a continuous memory block
			}
			else {
				cv::Rect roi(x - patch_width_half, y - patch_width_half, patch_width_half * 2, patch_width_half * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
				roi_img = gray_image(roi).clone(); // clone because we need a continuous memory block
			}
			roi_img.convertTo(roi_img, CV_32FC1); // vl_hog_put_image expects a float* (values 0.0f-255.0f)
			VlHog* hog = vl_hog_new(vlhog_variant, num_bins, false); // transposed (=col-major) = false
			vl_hog_put_image(hog, (float*)roi_img.data, roi_img.cols, roi_img.rows, 1, cell_size); // (the '1' is numChannels)
			int ww = static_cast<int>(vl_hog_get_width(hog)); // assert ww == hh == numCells
			int hh = static_cast<int>(vl_hog_get_height(hog));
			int dd = static_cast<int>(vl_hog_get_dimension(hog)); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
			Mat hogArray(1, ww*hh*dd, CV_32FC1); // safer & same result. Don't use C-style memory management.
			vl_hog_extract(hog, hogArray.ptr<float>(0));
			vl_hog_delete(hog);
			Mat hogDescriptor(hh*ww*dd, 1, CV_32FC1);
			// Stack the third dimensions of the HOG descriptor of this patch one after each other in a column-vector:
			for (int j = 0; j < dd; ++j) {
				Mat hogFeatures(hh, ww, CV_32FC1, hogArray.ptr<float>(0) + j*ww*hh); // Creates the same array as in Matlab. I might have to check this again if hh!=ww (non-square)
				hogFeatures = hogFeatures.t(); // necessary because the Matlab reshape() takes column-wise from the matrix while the OpenCV reshape() takes row-wise.
				hogFeatures = hogFeatures.reshape(0, hh*ww); // make it to a column-vector
				Mat currentDimSubMat = hogDescriptor.rowRange(j*ww*hh, j*ww*hh + ww*hh);
				hogFeatures.copyTo(currentDimSubMat);
			}
			hogDescriptor = hogDescriptor.t(); // now a row-vector转换为向量
			hog_descriptors.push_back(hogDescriptor);
		}
		// concatenate all the descriptors for this sample vertically (into a row-vector):
		hog_descriptors = hog_descriptors.reshape(0, hog_descriptors.cols * num_landmarks).t();
		return hog_descriptors;
	};

private:
	vector<std::string> imageFiles;
	Mat images;
	VlHogVariant vlhog_variant;
	int num_cells;
	int cell_size;
	int num_bins;
	int nType;
};

/**
 * Load the pre-calculated landmarks mean from the filesystem.
 *
 * @param[in] filename Path to a file with mean landmarks.
 * @return A cv::Mat of the loaded mean model.
 */
cv::Mat load_mean(fs::path filename)
{
	std::ifstream file(filename.string());
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open file: " + filename.string()));
	}

	string line;
	getline(file, line);

	vector<string> values;
	boost::split(values, line, boost::is_any_of(","));

	//int twice_num_landmarks = static_cast<int>(values.size());
	int num_landmarks = 68;
	Mat mean(1, 2 * num_landmarks, CV_32FC1);

	// For the sake of a brief example, we only use the
	// landmarks 31, 37, 46, 49 and 55 for training:
	// (we subtract 1 because std::vector's indexing starts at 0, ibug starts at 1)
	for (int i=0; i < num_landmarks; i++)
	{
		mean.at<float>(i) = std::stof(values[i]);
		mean.at<float>(i+num_landmarks) = std::stof(values[i + num_landmarks]); // y coordinates
	}
	// mean.at<float>(0) = std::stof(values[30]); // the x coordinates //取五个点
	// mean.at<float>(1) = std::stof(values[36]);
	// mean.at<float>(2) = std::stof(values[45]);
	// mean.at<float>(3) = std::stof(values[48]);
	// mean.at<float>(4) = std::stof(values[54]);
	// mean.at<float>(5) = std::stof(values[30 + 68]); // y coordinates
	// mean.at<float>(6) = std::stof(values[36 + 68]);
	// mean.at<float>(7) = std::stof(values[45 + 68]);
	// mean.at<float>(8) = std::stof(values[48 + 68]);
	// mean.at<float>(9) = std::stof(values[54 + 68]);

	return mean;
};

/**
 * Performs an initial alignment of the model, by putting the mean model into
 * the center of the face box.
 *
 * An optional scaling and translation parameters can be given to generate
 * perturbations of the initialisation.
 *
 * @param[in] mean Mean model points.
 * @param[in] facebox A facebox to align the model to.
 * @param[in] scaling_x Optional scaling in x of the model.
 * @param[in] scaling_y Optional scaling in y of the model.
 * @param[in] translation_x Optional translation in x of the model.
 * @param[in] translation_y Optional translation in y of the model.
 * @return A cv::Mat of the aligned points.
 */
Mat align_mean(Mat mean, cv::Rect facebox, float scaling_x=1.0f, float scaling_y=1.0f, float translation_x=0.0f, float translation_y=0.0f)
{
	// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
	// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
	// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
	Mat aligned_mean = mean.clone();
	Mat aligned_mean_x = aligned_mean.colRange(0, aligned_mean.cols / 2); //取点的x值
	Mat aligned_mean_y = aligned_mean.colRange(aligned_mean.cols / 2, aligned_mean.cols); //取关键点的y值
	aligned_mean_x = (aligned_mean_x*scaling_x + 0.5f + translation_x) * facebox.width + facebox.x;
	aligned_mean_y = (aligned_mean_y*scaling_y + 0.5f + translation_y) * facebox.height + facebox.y;
	return aligned_mean;
}


/**
 * Draws the given landmarks into the image.
 *
 * @param[in] image An image to draw into.
 * @param[in] landmarks The landmarks to draw.
 * @param[in] color Color of the landmarks to be drawn.
 */
void draw_landmarks(cv::Mat image, cv::Mat landmarks, cv::Scalar color = cv::Scalar(0.0, 255.0, 0.0))
{
	auto num_landmarks = std::max(landmarks.cols, landmarks.rows) / 2;
	for (int i = 0; i < num_landmarks; ++i) {
		cv::circle(image, cv::Point2f(landmarks.at<float>(i), landmarks.at<float>(i + num_landmarks)), 2, color);
	}
}


Mat ProjectShape(const Mat& landmark, cv::Rect& bounding_box){
	double centroid_x = bounding_box.x + bounding_box.width/2.0;
    double centroid_y = bounding_box.y + bounding_box.height/2.0; 
	int num_landmarks = 68;
    Mat temp(1, 2 * num_landmarks, CV_32FC1);
    for(int j = 0;j < num_landmarks;j++){
        temp.at<float>(j) = (landmark.at<float>(j)-centroid_x) / (bounding_box.width / 2.0);
         temp.at<float>(j + num_landmarks) = (landmark.at<float>(j + num_landmarks)-centroid_y) / (bounding_box.height / 2.0);  
    } 
    return temp;  
}

Mat ReProjectShape(const Mat& landmark, cv::Rect& bounding_box){


	double centroid_x = bounding_box.x + bounding_box.width/2.0;
    double centroid_y = bounding_box.y + bounding_box.height/2.0; 


	Mat aligned_mean = landmark.clone();
	Mat aligned_mean_x = aligned_mean.colRange(0, aligned_mean.cols / 2); //取点的x值
	Mat aligned_mean_y = aligned_mean.colRange(aligned_mean.cols / 2, aligned_mean.cols); //取关键点的y值
	aligned_mean_x = aligned_mean_x * bounding_box.width / 2.0 + centroid_x;
	aligned_mean_y = aligned_mean_y * bounding_box.height / 2.0 + centroid_y;

    return aligned_mean; 
}


Mat GetMeanShape(const Mat& shapes, vector<cv::Rect>& bounding_box){
    Mat mean(1, 2 * 68, CV_32FC1);
    for(int i = 0;i < bounding_box.size();i++){
        mean = mean + ProjectShape(shapes.row(i),bounding_box[i]);
    }
    mean = 1.0 / bounding_box.size() * mean;

    return mean;
}




/**
 * This app demonstrates learning of the descent direction from data for
 * a simple facial landmark detection sample app.
 *
 * First, we detect a face using OpenCV's face detector, and put the mean
 * landmarks into the face box. Then, the update step of the landmark coordinates
 * is learned using cascaded regression. HoG features are extracted around the
 * landmark positions, and from that, the update step towards the ground truth
 * positions is learned.
 *
 * This is an example of the library when a known template \c y is not available
 * during testing (because the HoG features are different for every subject).
 * Learning is thus performed without a \c y.
 */// ./examples/landmark_detection -d ../data/lfpw/trainset -m ../examples/data/mean_ibug_lfpw_68.txt -f ../examples/data/haarcascade_frontalface_alt.xml
int main(int argc, char *argv[])
{
	//fs  po 两个boost的命名空间
	fs::path trainingset, meanfile, facedetector;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("data,d", po::value<fs::path>(&trainingset)->required()->default_value("data/examples/ibug_lfpw_trainset"),
				"path to ibug LFPW example images and landmarks")
			("mean,m", po::value<fs::path>(&meanfile)->required()->default_value("data/examples/mean_ibug_lfpw_68.txt"),
				"pre-calculated mean from ibug LFPW")
			("facedetector,f", po::value<fs::path>(&facedetector)->required(),
				"full path to OpenCV's face detector (haarcascade_frontalface_alt2.xml)")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: landmark_detection [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	vector<std::string> training_images;
	Mat training_landmarks;
	vector<cv::Rect> boxList;
	try	{
		std::tie(training_images, training_landmarks) = load_ibug_data(trainingset,boxList); //load train data
	}
	catch (const fs::filesystem_error& e)
	{
		cout << e.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the pre-calculated (and scaled) mean of all landmarks:
	//Mat model_mean = load_mean(meanfile);

	Mat model_mean = GetMeanShape(training_landmarks,boxList); 
	// Load the face detector from OpenCV:
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load(facedetector.string()))//人脸检测模型
	{
		cout << "Error loading face detection model." << endl;
		return EXIT_FAILURE;
	}
	
	// Run the face detector and obtain the initial estimate x0 using the mean landmarks:
	Mat x0;
	for (size_t i = 0; i < training_images.size(); ++i) {

		x0.push_back(ReProjectShape(model_mean,boxList[i]));
		
	}

	// We might want to augment the training set by perturbing the
	// initialisations, which we skip here.

	// Create 3 regularised linear regressors in series:
	vector<LinearRegressor<>> regressors;
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 0.1f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 0.1f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 0.1f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 0.1f, true)));
	SupervisedDescentOptimiser<LinearRegressor<>> supervised_descent_model(regressors);
	
	HogTransform hog(training_images,1, VlHogVariant::VlHogVariantUoctti, 3 /*numCells*/, 12 /*cellSize*/, 4 /*numBins*/);

	// Train the model. We'll also specify an optional callback function:
	cout << "Training the model, printing the residual after each learned regressor: " << endl;
	auto print_residual = [&training_landmarks](const cv::Mat& current_predictions) {
		cout << "Current training residual: ";
		cout << cv::norm(current_predictions, training_landmarks, cv::NORM_L2) / cv::norm(training_landmarks, cv::NORM_L2) << endl;
	};
	
	supervised_descent_model.train(training_landmarks, x0, Mat(), hog, print_residual);

	// To test on a whole bunch of images, we could do something roughly like this:
	// supervisedDescentModel.test(x0_ts,
	//	Mat(),
	//	HogTransform(testImages, VlHogVariant::VlHogVariantUoctti, 3, 12, 4),
	//	[&groundtruth_ts](const cv::Mat& currentPredictions) { std::cout << cv::norm(currentPredictions, groundtruth_ts, cv::NORM_L2) / cv::norm(groundtruth_ts, cv::NORM_L2) << std::endl; }
	// );
	
	// Detect the landmarks on a single image:
	Mat image = cv::imread(training_images[1]);
	vector<cv::Rect> detected_faces;
	face_cascade.detectMultiScale(image, detected_faces, 1.2, 2, 0, cv::Size(50, 50));
	Mat initial_alignment = align_mean(model_mean, cv::Rect(detected_faces[0]));
	Mat prediction = supervised_descent_model.predict(initial_alignment, Mat(), HogTransform({ training_images[1] }, 1, VlHogVariant::VlHogVariantUoctti, 3, 12, 4));
	draw_landmarks(image, prediction, { 0, 0, 255 });
	cv::imwrite("out.png", image);
	cout << "Ran the trained model on an image and saved the result to out.png." << endl;
	// Save the learned model:
	std::ofstream learned_model_file("landmark_regressor_ibug_5lms.bin", std::ios::binary);
	cereal::BinaryOutputArchive output_archive(learned_model_file);
	output_archive(supervised_descent_model);

	
    cv::VideoCapture mCamera(0);
    if(!mCamera.isOpened()){
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    for(;;){
        mCamera >> image;
	    vector<cv::Rect> detected_faces;
	    face_cascade.detectMultiScale(image, detected_faces, 1.2, 2, 0, cv::Size(50, 50));
        if (detected_faces.size() == 0)
        {
            continue;
        }
       
	    Mat initial_alignment = align_mean(model_mean, cv::Rect(detected_faces[0]));
	    Mat prediction = supervised_descent_model.predict(initial_alignment, Mat(), HogTransform(image, 0,VlHogVariant::VlHogVariantUoctti, 3, 12, 4));

        draw_landmarks(image, prediction, { 0, 0, 255 });

        cv::imshow("Camera", image);
        if(27 == cv::waitKey(5)){
            mCamera.release();
            cv::destroyAllWindows();
            break;
        }
    }



	return EXIT_SUCCESS;
}
