 
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> matches;
    std::vector<float> euclidDist;
    
    for(auto match : kptMatches)
    {
        cv::KeyPoint preKeyPoint = kptsPrev[match.queryIdx];
        cv::KeyPoint curKeyPoint = kptsCurr[match.trainIdx];
        
        
        if(boundingBox.roi.contains(curKeyPoint.pt))
        {
            //float distPreCur = sqrt(pow((preKeyPoint.pt.y - curKeyPoint.pt.y),2) + pow((preKeyPoint.pt.x - curKeyPoint.pt.x),2));
            float distPreCur = cv::norm(curKeyPoint.pt - preKeyPoint.pt);
            euclidDist.push_back(distPreCur);
            matches.push_back(match);
        }
    }
    
    
    float medianDist, medianDistQ1, medianDistQ3;
   
    std::vector<float> euclidDistSort = euclidDist;   
    std::sort(euclidDistSort.begin(),euclidDistSort.end());

    int idx = floor(euclidDistSort.size()/2);
    int idxLow = ceil(euclidDistSort.size()*0.25);
    int idxHigh = ceil(euclidDistSort.size()*0.75);
    
    medianDist =euclidDistSort[idx];
    medianDistQ1 =euclidDistSort[idxLow];
    medianDistQ3 =euclidDistSort[idxHigh];

    float medianRange = medianDistQ3-medianDistQ1;
        
    for(int i = 0; i< euclidDist.size(); ++i)
    {
        if((euclidDist[i] > medianDistQ1-1.5*medianRange) && (euclidDist[i] < medianDistQ3 + 1.5*medianRange))
        {
            boundingBox.keypoints.push_back(kptsCurr[matches[i].trainIdx]);
            boundingBox.kptMatches.push_back(matches[i]);
        }
    }

}


void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)

{   
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1.0 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    /*
    cout << " " << endl;
    cout << " TTC Camera: " << TTC << endl;
    cout << " " << endl;
   */
}


void filtOutliersMedian(std::vector<LidarPoint> &inputPoints, std::vector<LidarPoint> &filtPoints)
{
    
    std::vector<double> inputSortX;
    std::vector<double> inputSortY;
    
    for (auto point : inputPoints)
    {
        inputSortX.push_back(point.x);
        inputSortY.push_back(point.y);
    }


    sort(inputSortX.begin(),inputSortX.end());
    sort(inputSortY.begin(),inputSortY.end());
    

    int idx = floor(inputPoints.size()/2);
    int idxLow = ceil(inputPoints.size()*0.25);
    int idxHigh = ceil(inputPoints.size()*0.75);

    //float medX = inputSortX[idx];
    float medLowX= inputSortX[idxLow];
    float medLowY= inputSortY[idxLow];
    float medHighX = inputSortX[idxHigh];
    float medHighY = inputSortY[idxHigh];
    float medRangeX = medHighX - medLowX;
    float medRangeY = medHighY - medLowY;

    for(auto point : inputPoints)
    {
        //if ((point.x > medLowX - 1.5*medRangeX) && (point.x < medHighX + 1.5*medRangeX) && (point.y > medLowY - 1.5*medRangeY) && (point.y < medHighY + 1.5*medRangeY))
        if ((point.x > medLowX - 1.5*medRangeX) && (point.x < medHighX + 1.5*medRangeX))
        {
            filtPoints.push_back(point);
        }
    }
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    
    // Filter LidarPoints:
    std::vector<LidarPoint> filtLidarPointsPre;
    filtOutliersMedian(lidarPointsPrev, filtLidarPointsPre);
    std::vector<LidarPoint> filtLidarPointsCur;
    filtOutliersMedian(lidarPointsCurr, filtLidarPointsCur);

    // auxiliary variables
    double dT = 1.0/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    //for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    for (auto it = filtLidarPointsPre.begin(); it != filtLidarPointsPre.end(); ++it)
    {
        
        //if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
    }

    //for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    for (auto it = filtLidarPointsCur.begin(); it != filtLidarPointsCur.end(); ++it)
    {
        
        //if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
     
    /*
    cout << " " << endl;
    cout << " TTC Lidar: " << TTC << "minXCurr : " << minXCurr <<  "minXPrev : "<< minXPrev << endl;
    cout << " " << endl;
    */
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
   
    for(auto preBB: prevFrame.boundingBoxes)
    {
        std::map<int, int> idMatches;  // Create the map with (boxID_prev, boxID_curr) we need to see which box has the most of this
        for(auto curBB : currFrame.boundingBoxes)
        {
            //currBox.kptMatches = currFrame.kptMatches;
            for(auto match: matches)
            {
                auto preKeyPoint = prevFrame.keypoints[match.queryIdx];
                auto curKeyPoint = currFrame.keypoints[match.trainIdx];

                if(preBB.roi.contains(preKeyPoint.pt) && curBB.roi.contains(curKeyPoint.pt))
                {
                    if (idMatches.count(curBB.boxID)==0)
                    {
                        idMatches[curBB.boxID] =1;
                    }
                    else
                    {
                        idMatches[curBB.boxID]++;
                    }           
                }
            }
        }

        int preMax = 0;
        int curMax = 0;
        for(auto it = idMatches.begin(); it!=idMatches.end(); ++it)
        {
            if(it->second > curMax)
            {
                preMax = it->first;
                curMax = it->second;
            }
        }
        bbBestMatches[preBB.boxID] = preMax;
    }
    
    /*
    for (auto bb : bbBestMatches)
    {
        cout << "fist element : " << bb.first << "; second element : " << bb.second << endl;
    }
    */
    //std::cout<<"prevBoxMatch:" << bbBestMatches[0]<<", currBoxMatch:" <<bbBestMatches[1] <<endl;
}
