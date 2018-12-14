#include <vector>
#include <iostream>
#include <fstream>
#include "cereal/archives/binary.hpp"
#include "superviseddescent/superviseddescent.hpp"
#include "superviseddescent/regressors.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
extern "C" {
	#include "hog.h" // From the VLFeat C library
}

using namespace std;
using namespace cv;

void draw_landmarks(cv::Mat image, cv::Mat landmarks, cv::Scalar color = cv::Scalar(0.0, 255.0, 0.0))
{
	auto num_landmarks = std::max(landmarks.cols, landmarks.rows) / 2;
	for (int i = 0; i < num_landmarks; ++i) {
		cv::circle(image, cv::Point2f(landmarks.at<float>(i), landmarks.at<float>(i + num_landmarks)), 2, color);
	}
}
bool load_ldmarkmodel(std::string filename, SupervisedDescentOptimiser<LinearRegressor<>> &model)
{
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open())
        return false;
    cereal::BinaryInputArchive input_archive(file);
    input_archive(model);
    file.close();
    return true;
}

int main()
{
	/*********************
    std::vector<ImageLabel> mImageLabels;
    if(!load_ImageLabels("mImageLabels-test.bin", mImageLabels)){
        mImageLabels.clear();
        ReadLabelsFromFile(mImageLabels, "labels_ibug_300W_test.xml");
        save_ImageLabels(mImageLabels, "mImageLabels-test.bin");
    }
    std::cout << "��������һ����: " <<  mImageLabels.size() << std::endl;
	*******************/
	vector<LinearRegressor<>> regressors;
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 0.1f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 0.1f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 0.1f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 0.1f, true)));
    SupervisedDescentOptimiser<LinearRegressor<>> supervised_descent_model(regressors);
    std::string modelFilePath = "landmark_regressor_ibug_5lms.bin";
    while(!load_ldmarkmodel(modelFilePath, supervised_descent_model)){
        std::cout << "�ļ��򿪴��������������ļ�·��." << std::endl;
        std::cin >> modelFilePath;
    }

    cv::VideoCapture mCamera(0);
    if(!mCamera.isOpened()){
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    cv::Mat Image;
    cv::Mat current_shape;
    for(;;){
        mCamera >> Image;
	    vector<cv::Rect> detected_faces;
	    face_cascade.detectMultiScale(image, detected_faces, 1.2, 2, 0, cv::Size(50, 50));
	    Mat initial_alignment = align_mean(model_mean, cv::Rect(detected_faces[0]));
	    Mat prediction = supervised_descent_model.predict(initial_alignment, Mat(), HogTransform({ image }, VlHogVariant::VlHogVariantUoctti, 3, 12, 4));
	    draw_landmarks(image, prediction, { 0, 0, 255 });


        int numLandmarks = current_shape.cols/2;
        for(int j=0; j<numLandmarks; j++){
            int x = current_shape.at<float>(j);
            int y = current_shape.at<float>(j + numLandmarks);
            std::stringstream ss;
            ss << j;
//            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
            cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("Camera", Image);
        if(27 == cv::waitKey(5)){
            mCamera.release();
            cv::destroyAllWindows();
            break;
        }
    }

    system("pause");
    return 0;
}






















