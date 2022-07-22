#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = true;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    int normType;
    // without SIFT
    if (matcherType.compare("MAT_BF") == 0)
    {
        if (descriptorType.compare("DES_BINARY")==0)
        {
            normType = cv::NORM_HAMMING;
        }
        else
        {
            normType = cv::NORM_L2;
        }
               
        double t = (double)cv::getTickCount();
        matcher = cv::BFMatcher::create(normType, crossCheck);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
        //cout << matcherType << " needs " << 1000 * t / 1.0 << " ms" << endl;
    }
    // For SIFT
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        
        if (descSource.type() != CV_32F) 
        {
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F) 
        {
            descRef.convertTo(descRef, CV_32F);
        }

        double t = (double)cv::getTickCount();
        matcher = cv::FlannBasedMatcher::create();
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
        //cout << matcherType << " needs " << 1000 * t / 1.0 << " ms" << endl;
    }


    // perform matching task
    double t = (double)cv::getTickCount();
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match) 
        matcher->match(descSource, descRef, matches);
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        int k = 2;
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        // Filter matches using descriptor distance ratio test
        double distRatio = 0.8;
        for (auto point : knn_matches) 
        {
            if ( 2 == point.size() && (point[0].distance < distRatio * point[1].distance) ) 
            {
                matches.push_back(point[0]);
            }
        }
    }
         t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
        //cout << selectorType << " selecting best match in " << 1000 * t / 1.0 << " ms" << endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
//void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &descriptorTime)
{
    double t_des = (double)cv::getTickCount();
    // select appropriate descriptor
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(threshold, octaves, patternScale);
         extractor->compute(img, keypoints, descriptors);
    }
    else if  (descriptorType.compare("BRIEF") == 0)
    {
       //extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
       cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        extractor->compute(img, keypoints, descriptors);
    }
    else if  (descriptorType.compare("ORB") == 0)
    {
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
         extractor->compute(img, keypoints, descriptors);
    }
    else if  (descriptorType.compare("FREAK") == 0)
    {
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::FREAK::create();
         extractor->compute(img, keypoints, descriptors);
    }
    else if  (descriptorType.compare("AKAZE") == 0)
    {
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::AKAZE::create();
         extractor->compute(img, keypoints, descriptors);
    }
    else if  (descriptorType.compare("SIFT") == 0)
    {
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SIFT::create();
         extractor->compute(img, keypoints, descriptors);
    }

    // perform feature description
    t_des = ((double)cv::getTickCount() - t_des) / cv::getTickFrequency();
    //cout << descriptorType << " descriptor extraction in " << 1000 * t_des / 1.0 << " ms" << endl;
    descriptorTime = t_des;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
//void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis,double &detectorTime)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t_detShi = (double)cv::getTickCount();
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
    t_detShi = ((double)cv::getTickCount() - t_detShi) / cv::getTickFrequency();
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t_detShi / 1.0 << " ms" << endl;
    detectorTime = t_detShi;
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

//void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, double &detectorTime)
{
// compute detector parameters based on image size
    int blockSize = 4;        //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int apertureSize = 3;     //  Sobel Operator
    int minResponse = 100;    // min value for a corner
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double k = 0.04;

    // Apply corner detection
    double t_detHar = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img ,dst ,blockSize ,apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm ,dst_norm_scaled);
    
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }    

    t_detHar = ((double)cv::getTickCount() - t_detHar) / cv::getTickFrequency();
    //cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t_detHar / 1.0 << " ms" << endl;
    detectorTime = t_detHar;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

//void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis, double &detectorTime)
{
    double t_det = (double)cv::getTickCount(); 

    if (detectorType.compare("FAST")==0)
    {  
        cv::Ptr<cv::Feature2D> detector;
        detector = cv::FastFeatureDetector::create();
        detector-> detect(img, keypoints);
        
        // visualize results
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "FAST Corner Detector Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        } 
    }       
    else if (detectorType.compare("BRISK")==0)
    {
        cv::Ptr<cv::Feature2D> detector;
        detector = cv::BRISK::create();        
        detector-> detect(img, keypoints);
        
        // visualize results
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "BRISK Corner Detector Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }  
    }
    else if (detectorType.compare("ORB")==0)
    {
        cv::Ptr<cv::Feature2D> detector;
        detector = cv::ORB::create();
        detector-> detect(img,keypoints);

            // visualize results
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "ORB Corner Detector Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        } 
    }
    else if (detectorType.compare("AKAZE")==0)
    {
        cv::Ptr<cv::Feature2D> detector;
        detector = cv::AKAZE::create();
        detector->detect(img,keypoints);

        // visualize results
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "AKAZE Corner Detector Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }
    }
    else if (detectorType.compare("SIFT")==0)
    {
        cv::Ptr<cv::xfeatures2d::SIFT> detector;
        detector = cv::xfeatures2d::SIFT::create();
        detector-> detect(img, keypoints);
        //cout << "Detctor Type = " << detectorType << endl; 
        
        // visualize results
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "SIFT Corner Detector Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);;
        }
    }

    t_det = ((double)cv::getTickCount() - t_det) / cv::getTickFrequency();
    //cout << detectorType << " with n=" << keypoints.size() << " keypoints in " << 1000 * t_det / 1.0 << " ms" << endl; 
    detectorTime = t_det;   
}
