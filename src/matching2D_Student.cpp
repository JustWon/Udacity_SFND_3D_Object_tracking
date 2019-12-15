#include <numeric>
#include "matching2D.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher->knnMatch( descSource, descRef, knn_matches, 2 );

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.8f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        matches = good_matches;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 5;
    int apertureSize = 5;
    double k = 0.01;
    double thresh = 0.1;
    double t = (double)cv::getTickCount();
    cv::Mat dst = cv::Mat::zeros( img.size(), CV_32FC1 );

    cv::cornerHarris( img, dst, blockSize, apertureSize, k );

    cv::Mat dst_norm;
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );

    for( int i = 0; i < dst.rows ; i++ )
    {
        for( int j = 0; j < dst.cols; j++ )
        {
            // printf("%f\n", dst.at<float>(i,j));
            if( dst.at<float>(i,j) > thresh )
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = blockSize;
                keypoints.push_back(newKeyPoint);
            }
        }
    }

    cv::namedWindow( "corners_window" );
    cv::imshow( "corners_window", dst_norm );

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // double t = (double)cv::getTickCount();

    // cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create(1000, 0.1, 1, 3, 5, true, 0.4);
    // gftt->setHarrisDetector(true);

    // t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsFast(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    const cv::Ptr<cv::Feature2D>& detector  = cv::FastFeatureDetector::create(100);
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsBrisk(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    const cv::Ptr<cv::Feature2D>& detector  = cv::BRISK::create();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsORB(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    const cv::Ptr<cv::Feature2D>& detector  = cv::ORB::create();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsAKAZE(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    const cv::Ptr<cv::Feature2D>& detector  = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_KAZE);
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsSIFT(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    const cv::Ptr<cv::Feature2D>& detector = cv::xfeatures2d::SIFT::create();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

