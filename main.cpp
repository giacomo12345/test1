/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO*/

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/base.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>


using namespace cv;
using namespace std;

vector<DMatch> autotune_matches(vector<DMatch> Matches, float min_dist, float ratio)
{
	vector<DMatch> match;
	for (int j = 0; j < int(Matches.size()); j++)
	{
		if (Matches[j].distance < min_dist * ratio)
			match.push_back(Matches[j]);
	}
	return match;
}

void draw_rectangle(Mat img_object, Mat img_matches, Mat H)
{
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)img_object.cols, 0);
	obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
	obj_corners[3] = Point2f(0, (float)img_object.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
		scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
		scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
		scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
		scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
	//-- Show detected matches
	namedWindow("Good Matches & Object detection", WINDOW_AUTOSIZE);
	resize(img_matches, img_matches, Size(img_matches.cols / 2, img_matches.rows / 2));
	imshow("Good Matches & Object detection", img_matches);



}



//-----------------------------------------------------functions
/*
void compute_match(vector<Mat> Descriptors, vector<vector<DMatch>>& Matches)
   {

	vector<DMatch> tmp_matches;

	// 4 should be the norm-type ENUM that correspond to NORM_HAMMING --- we use also cross-match
	Ptr<BFMatcher> matcher = BFMatcher::create(4, true);

	for (int i = 0; i < Descriptors.size() - 1; i++)
	{
		//cout << "Size of " << i << "-th descriptor: " << Descriptors[i].size() << endl;

		matcher->match(Descriptors[i], Descriptors[i + 1], tmp_matches, Mat());

		Matches.push_back(tmp_matches);
		cout << "Match between " << i + 1 << "-" << i + 2 << " computed." << endl;
	}

	// Add last-first match
	matcher->match(Descriptors.back(), Descriptors[0], tmp_matches, Mat());
	Matches.push_back(tmp_matches);
	cout << "Match between last-first computed." << endl;
	return;
}
*/
//-----------------------------------------------------variables




//------------------------------------------------------------------------------------
//                               MAIN FUNCTION
//------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {

	/* create a vector of images to find and track and a vector for the video frames*/
	std::vector<cv::Mat> src;
	std::vector<cv::Mat> frames;

	/* upload images to track */
	// ADJUST THIS CHECK....WE CAN HAVE 1..2...3..4..5.. IMAGES...
	// FIND A WAY TO KNOW HOW MANY IMAGES ARE PASSED THRUOGHT THE COMMAND LINE
	// AND USE THIS NUMBER AS SUPERIOR LIMIT TO THE FOR CYCLE BELOW
	if (argc < 5) {
		std::cout << "Error - 4 images needed" << std::endl;
		return 0;
	}

	for (int i = 1; i < 5; i++) {
		Mat img = imread(argv[i], IMREAD_COLOR);
		src.push_back(img);
		resize(img, img, Size(img.cols / 2, img.rows / 2));
		imshow("source images", img);
		waitKey(100);
	}
	destroyWindow("source images");




	/* upload video frames */
	int count = 0;
	cout << "frames uploading";
	cv::VideoCapture cap("video.mov");
	if (cap.isOpened())
	{
		for (;;) {


			cv::Mat frame;
			cap >> frame;
			if (count == 2) break;
			if (!cap.read(frame)) break;
			frames.push_back(frame);
			if (count % 100 == 0) std::cout << ".";

			/*namedWindow("video", WINDOW_FREERATIO);
			imshow("video", frame);
			*/
			waitKey(1);
			count++;

		}
	}
	else std::cout << "error 404 video not found" << std::endl;
	cap.release();
	//destroyWindow("video");

	cout << "number frames found: " << frames.size() << std::endl;

	/* get the first frame to locate features */
	Mat mainFrame = frames[0];
	Mat tmp = mainFrame;
	src.push_back(tmp);
	namedWindow("main frame", WINDOW_FREERATIO);
	imshow("main frame", mainFrame);
	cout << "feature detection: " << std::endl;

	//create the objects detector and extractor for SIFT 
	cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
	cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();

	//variables to store the results of SIFT processing
	std::vector < std::vector<cv::KeyPoint> > keypoints;
	std::vector<cv::Mat> descriptors;

	//String indice;              ho tenuto queste due righe perche' non so se e' piu' corretto tenerle fuori dal ciclo for per inizializzarle una sola volta o tenerle(a capo)
	//Mat tmpdescriptors;		  dentro il ciclo for perche' in realta' sarebbero variabili del ciclo e quindi e' meglio tenerle dentro anche se le inizializza ad ogni iterazione.

	/* array of colors */
	std::vector <cv::Scalar> colors;
	for (int i = 0; i < 5; i++) {
		Scalar clr(rand() % 255, rand() % 255, rand() % 255);
		colors.push_back(clr);
	}

	//ciclo for per determinare keypoints e relativi descriptors dei quattro libri presi singolarmente e quando sono tutti insieme nel mainFrame
	for (int i = 0; i < 5; i++)
	{
		String indice = to_string(i);

		Mat input = src[i];
		std::vector<cv::KeyPoint> keypoints_tmp;
		detector->detect(input, keypoints_tmp);
		keypoints.push_back(keypoints_tmp);

		cv::Mat output;
		cv::drawKeypoints(input, keypoints[i], output, colors[i]);
		//cv::imshow("image" + indice, output);
		waitKey(100);


		Mat tmpdescriptors;
		extractor->detectAndCompute(input, Mat(), keypoints[i], tmpdescriptors);
		descriptors.push_back(tmpdescriptors);
		//imshow("ff" + indice, descriptors[i]);


	}



	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
	std::vector< std::vector<DMatch> > matches;

	for (int i = 0; i < 4; i++)
	{
		vector<DMatch> tmp_matches;
		matcher->match(descriptors[i], descriptors[4], tmp_matches, Mat());
		matches.push_back(tmp_matches);
	}

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	//keypoints_object = keypoints[0];
	keypoints_scene = keypoints[4];
	//Mat img_object = src[0].clone();
	Mat img_scene = src[4].clone();;
	std::vector<cv::Mat> img_matches;

	std::vector<std::vector<DMatch>> good_matches;
	for (int i = 0; i < matches.size(); i++)
	{
		String indice = to_string(i);
		float ratio = 1.5;
		float instance_ratio = ratio;

		std::vector<DMatch> tmp_good_matches;
		// push in bestMatches the top 50 matched features
		int nbMatch = int(matches[i].size());
		Mat tab(nbMatch, 1, CV_32F);

		float dist;
		float min_dist = -1.;

		// Find the minumun distance between matchpoints
		for (int j = 0; j < nbMatch; j++)
		{
			dist = matches[i][j].distance;

			// update the minumun distance
			if (min_dist < 0 || dist < min_dist)
				min_dist = dist;
		}


		// Adapt the ratio in order to get at least 120 matches per couple of adjacent images
		do
		{
			tmp_good_matches = autotune_matches(matches[i], min_dist, instance_ratio);
			instance_ratio = 2 * instance_ratio;
		} while (tmp_good_matches.size() < 120);

		good_matches.push_back(tmp_good_matches);

		//cout << "Size of " << i << "-th" << "TOP matches: " << BestMatches[i].size() << endl;




		//-- Draw matches
		cv::Mat tmp_img_matches;


		drawMatches(src[i], keypoints[i], img_scene, keypoints_scene, good_matches[i], tmp_img_matches, colors[i],
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		img_matches.push_back(tmp_img_matches);

		namedWindow("Good Matches" + indice, WINDOW_AUTOSIZE);
		resize(img_matches[i], img_matches[i], Size(img_matches[i].cols / 2, img_matches[i].rows / 2));

		imshow("Good Matches" + indice, img_matches[i]);

	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	std::vector<cv::Mat> H;
	for (int i = 0; i < good_matches.size(); i++)
	{
		cout << "ok" << endl;
		String indice = to_string(i);
		for (vector<DMatch>::iterator it = good_matches[i].begin(); it != good_matches[i].end() - 1; ++it)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints[i][good_matches[i][0].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i][0].trainIdx].pt);
		}
		cout << "siamo a buon punto" << endl;
		/*cout << "good_matches size:     " << good_matches[i].size() << std::endl;
		cout << "keypoints_object size:     " << keypoints[i].size() << std::endl;
		cout << "obj size:     " << obj.size() << std::endl;
		cout << "scene size:     " << scene.size() << std::endl;
		*/
		Mat tmp_H = findHomography(obj, scene, RANSAC);
		H.push_back(tmp_H);

		cout << "ancora poco" << endl;


	}

	cout << "oh laaaaaaaa" << endl;
	
	//for (int i = 0; i < 4; i++)
	//{
	draw_rectangle(src[0], img_matches[0], H[0]);
	//}
	cout << "oro" << endl;
	/*std::vector<std::vector<Point2f>> obj_corners;
	std::vector<Point2f>tmp_obj_corners;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			tmp_obj_corners.push_back(Point2f(0, 0));
			
		}
		obj_corners.push_back(tmp_obj_corners);
	}*/
	
	//-- Get the corners from the image_1 ( the object to be "detected" )
	/*for (int i = 0; i < 4; i++)
	{
		String indice = to_string(i);
		cout << "ci arriviamo" << endl;
		obj_corners[i][0] = Point2f(0, 0);
		obj_corners[i][1] = Point2f((float)src[i].cols, 0);
		obj_corners[i][2] = Point2f((float)src[i].cols, (float)src[i].rows);
		obj_corners[i][3] = Point2f(0, (float)src[i].rows);
		std::vector<Point2f> scene_corners(4);
		cout << "ancora un po'" << endl;
		perspectiveTransform(obj_corners[i], scene_corners, H[i]);
		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		cout << "ti supplico" << endl;
		line(img_matches[i], scene_corners[0] + Point2f((float)src[i].cols, 0),
			scene_corners[1] + Point2f((float)src[i].cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches[i], scene_corners[1] + Point2f((float)src[i].cols, 0),
			scene_corners[2] + Point2f((float)src[i].cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches[i], scene_corners[2] + Point2f((float)src[i].cols, 0),
			scene_corners[3] + Point2f((float)src[i].cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches[i], scene_corners[3] + Point2f((float)src[i].cols, 0),
			scene_corners[0] + Point2f((float)src[i].cols, 0), Scalar(0, 255, 0), 4);
		cout << "fatooooo" << endl;
		//-- Show detected matches
		namedWindow("Good Matches & Object detection", WINDOW_AUTOSIZE);
		resize(img_matches[i], img_matches[i], Size(img_matches[i].cols / 2, img_matches[i].rows / 2));
		imshow("Good Matches & Object detection" + indice, img_matches[i]);

	}*/

	


	cout << "END" << std::endl;



	waitKey(0);
	return 0;
}

//------------------------------------------------------------------------------------
//                                FUNCTIONS
//------------------------------------------------------------------------------------

	/*namedWindow("Still Better Good Matches" + indice, WINDOW_AUTOSIZE);
		resize(outputHomography[i], outputHomography[i], Size(outputHomography[i].cols / 2, outputHomography[i].rows / 2));
		imshow("Still Better Good Matches" + indice, outputHomography[i]);
		waitKey(100);*/


		/*for (int i = 0; i < 4; i++)
			{
				String indice = to_string(i);
				obj_corners[i][0] = Point2f(0, 0);
				obj_corners[i][1] = Point2f((float)src[i].cols, 0);
				obj_corners[i][2] = Point2f((float)src[i].cols, (float)src[i].rows);
				obj_corners[i][3] = Point2f(0, (float)src[i].rows);
				std::vector<Point2f> scene_corners(4);

				perspectiveTransform(obj_corners[i], scene_corners, H[i]);
				//-- Draw lines between the corners (the mapped object in the scene - image_2 )

				line(img_matches[i], scene_corners[0] + Point2f((float)src[i].cols, 0),
					scene_corners[1] + Point2f((float)src[i].cols, 0), Scalar(0, 255, 0), 4);
				line(img_matches[i], scene_corners[1] + Point2f((float)src[i].cols, 0),
					scene_corners[2] + Point2f((float)src[i].cols, 0), Scalar(0, 255, 0), 4);
				line(img_matches[i], scene_corners[2] + Point2f((float)src[i].cols, 0),
					scene_corners[3] + Point2f((float)src[i].cols, 0), Scalar(0, 255, 0), 4);
				line(img_matches[i], scene_corners[3] + Point2f((float)src[i].cols, 0),
					scene_corners[0] + Point2f((float)src[i].cols, 0), Scalar(0, 255, 0), 4);

				//-- Show detected matches
				namedWindow("Good Matches & Object detection", WINDOW_AUTOSIZE);
				resize(img_matches[i], img_matches[i], Size(img_matches[i].cols / 2, img_matches[i].rows / 2));
				imshow("Good Matches & Object detection" + indice, img_matches[i]);

			}
			*/