#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <string>

using namespace cv;
using namespace cv::ml;


void TraingOnePic(Mat &training_mat, String imgname, int file_num);

// Maching Learning Project using human face to train the model

int main(int, char**)
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	int labels[4] = { 1, 1, 1, -1 };
	Mat labelsMat(4, 1, CV_32SC1, labels);


	Mat training_mat(4, 187*181, CV_32FC1);

	TraingOnePic(training_mat, "face1.jpg", 0);
	TraingOnePic(training_mat, "face2.jpg", 1);
	TraingOnePic(training_mat, "face3.jpg", 2);
	TraingOnePic(training_mat, "dogface.jpg", 3);

	Ptr<TrainData> td = TrainData::create(training_mat, ROW_SAMPLE, labelsMat);

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::POLY);
	svm->setGamma(3);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	svm->train(td);

	Mat sampleMat = imread("face4.jpg", 0);
	Mat im_f;

	sampleMat.convertTo(im_f, CV_32F);

	float response = svm->predict(im_f.reshape(1, 1));
	printf("response:%f \n", response);	

	sampleMat = imread("dogface2.jpg", 0);
	sampleMat.convertTo(im_f, CV_32F);

	response = svm->predict(im_f.reshape(1, 1));
	printf("response:%f \n", response);

	// Or train the SVM with optimal parameters
	//svm->trainAuto(td);
	

	imshow("SVM Simple Example", image); // show it to the user*/
	waitKey(0);

}


void TraingOnePic(Mat &training_mat, String imgname, int file_num)
{
	Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
	int ii = 0; // Current column in training_mat

	for (int i = 0; i<img_mat.rows; i++) {
		for (int j = 0; j < img_mat.cols; j++) {
			training_mat.at<float>(file_num, ii++) = img_mat.at<uchar>(i, j);
		}
	}
}