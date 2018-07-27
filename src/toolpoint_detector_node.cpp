#include "ros/ros.h"
#include "geometry_msgs/Point.h"

#include <iostream>
#include <fstream>
#include <iomanip>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include "/opt/ros/kinetic/include/opencv-3.3.1-dev/opencv2/opencv.hpp"
//#include "/opt/ros/kinetic/include/opencv-3.3.1-dev/opencv2/highgui/highgui.hpp"

//#include "/usr/include/opencv2/opencv.hpp"
//#include "/usr/include/opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class centroidInfo {
public:
    centroidInfo(){
        setPos(0,0);
        setVel(0,0);
        setSeqID(0);
        setImgIndex(0);
        setVelChange(false);
    }
    ~centroidInfo(){}
    centroidInfo operator=(const centroidInfo & other) {
        this->setPos(other.getPos());
        this->setVel(other.getVel());
        this->setSeqID(other.getSeqID());
        this->setImgIndex(other.getImgIndex());
        this->setVelChange(other.getVelChange());
        return *this;
    }
    Point2f getPos() const {return (pos);}
    Point2f getVel() const {return (vel);}
    int getSeqID() const {return (seqID);}
    int getImgIndex() const {return (imgIndex);}
    bool getVelChange() const {return(velChange);}
    void setPos(const Point2f & p){pos = p;}
    void setVel(const Point2f& v){vel = v;}
    void setPos(const float & px, const float & py){pos.x = px; pos.y = py;}
    void setVel(const float & vx, const float & vy){vel.x = vx; vel.y = vy;}
    void setSeqID(const int & id){seqID = id;}
    void setImgIndex(const int & index){imgIndex = index;}
    void setVelChange(const bool & vc){velChange = vc;}
private:
    Point2f pos;
    Point2f vel;
    int seqID;
    int imgIndex;
    bool velChange;
};
class IPParameters{
public:
    IPParameters(){}
    IPParameters(const uint & ID) {
        //blue bounds
        Scalar minC, maxC;
        minC[0] = 0; minC[1] = 70;    minC[2] = 0; maxC[0] = 255; maxC[1] = 200;  maxC[2] = 20; // bgr
        setMinMaxColor(minC,maxC);
        switch(ID) {
        case 1:
            maskCorners.push_back(Point(0,480)); maskCorners.push_back(Point(0,450)); maskCorners.push_back(Point(450,480));
            maskHand=false;
            break;
        case 2:
            maskCorners.push_back(Point(0,480)); maskCorners.push_back(Point(0,450)); maskCorners.push_back(Point(450,480));
            maskHand=false;
            break;
        case 3:
            maskCorners.push_back(Point(0,480)); maskCorners.push_back(Point(0,450)); maskCorners.push_back(Point(450,480));
            maskHand=false;
            break;
        case 4:
            maskCorners.push_back(Point(0,480)); maskCorners.push_back(Point(0,450)); maskCorners.push_back(Point(450,480));
            maskHand=false;
            break;
        case 7:
            maskCorners.push_back(Point(110,480)); maskCorners.push_back(Point(140,470)); maskCorners.push_back(Point(400,475)); maskCorners.push_back(Point(640,475)); maskCorners.push_back(Point(640,480));
            maskHand=false;
            break;
        default:
            maskCorners.push_back(Point(0,0));  //no mask
            maskHand=false;
            break;
        }
    }
    ~IPParameters(){}
    vector<Point> getMaskCorners() const {return (maskCorners);}
    bool getMaskHand() const {return(maskHand);}
    void setMinMaxColor(const Scalar min, const Scalar max){min_Color=min; max_Color = max;}
    Scalar getMinColor() const {return(min_Color);}
    Scalar getMaxColor() const {return(max_Color);}
private:
    vector<Point> maskCorners;
    bool maskHand;
    Scalar min_Color;
    Scalar max_Color;
};
istream& operator>>(istream &cInput, centroidInfo &c){
    int id, index, vc;
    float px, py, vx, vy;
    cInput>> id;
    cInput>> index;
    cInput>> px;
    cInput>> py;
    cInput>> vx;
    cInput>> vy;
    cInput>> vc;
    c.setImgIndex(index);
    c.setSeqID(id);
    c.setPos(px, py);
    c.setVel(vx, vy);
    c.setVelChange((bool)(vc));
    return(cInput);
}
ostream& operator<<(ostream &cOutput, const centroidInfo &c){
    cOutput << c.getSeqID() << "\t" << c.getImgIndex() << "\t"
            << c.getPos().x << "\t" << c.getPos().y << "\t"
            << c.getVel().x << "\t" << c.getVel().y << "\t"
            << c.getVelChange() << "\t";
    return(cOutput);
}
vector<Point> GetLargestContour(const Mat & srcImg, const float & min_pixel_){
    //finding contours
    vector<Vec4i> hierarchy;
    Mat to_contour = srcImg.clone();
    vector< vector<Point> > contours;
    findContours(to_contour, contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE ); // Find the contours in the image
    //find largest
    vector<Point> largestContour;
    float mPixel = min_pixel_;
    for(unsigned int i = 0; i< contours.size(); i++ ) {// iterate through each contour.
        if(contourArea(contours[i],false) > mPixel){//the blob can be considered if it contains enough pixels
            largestContour = contours[i];
            mPixel = contourArea(contours[i],false);
        }
    }
    if(largestContour.size() == 0){//at least one contour needed !
        cout << "no largest contour found " << endl;
    }
    return(largestContour);
}
Mat MorphologyOperations(const Mat& srcImg, const int & filtSize, const uint & numOpeningAndClosing, const uint & numDilations){
    Mat dstImg = srcImg.clone();
    for (uint i=0; i < numOpeningAndClosing; i++) {
        //morphological opening (removes small objects from the foreground)
        erode(dstImg, dstImg, getStructuringElement(MORPH_ELLIPSE, Size(filtSize, filtSize)) );
        dilate(dstImg, dstImg, getStructuringElement(MORPH_ELLIPSE, Size(filtSize, filtSize)) );
        //morphological closing (removes small holes from the foreground)
        dilate(dstImg, dstImg, getStructuringElement(MORPH_ELLIPSE, Size(filtSize, filtSize)) );
        erode(dstImg, dstImg, getStructuringElement(MORPH_ELLIPSE, Size(filtSize, filtSize)) );
    }
    //a lot of dilations
    for (uint i =0; i < numDilations; i++) {
        dilate(dstImg, dstImg, getStructuringElement(MORPH_ELLIPSE, Size(filtSize, filtSize)) );
    }
    return(dstImg);
}
Mat LoadImage(const int & cntImg, const bool &rgb, const string & fName){//TODO provide just folder name as argument
    Mat image;
    ostringstream ossCnt, ossType;
    ossCnt << setw(4) << setfill('0') << cntImg;
    if (rgb) ossType.str("rgb");
    else ossType.str("depth");
    string imageName = fName + ossType.str() + ("/0") + ossCnt.str() + string(".png");
    cout << "==== BLOB TRACKER IMAGE " << imageName << " ====" << endl;
    image = imread(imageName);// Read the file//for depth maybe use CV_16U
    if (!rgb) image = 10*image;
    if(!image.data){
        cout <<  "Could not open or find the image" << endl ;
        exit(0);
    }
    return(image);
}
void writePathOnImage(Mat & img, const uint & Id, const uint & ImgNb){
    //load start and end image
    ostringstream ossID;
    ossID << setw(3) << setfill('0') << Id;
    ostringstream ossCnt;
    ossCnt << setw(4) << setfill('0') << ImgNb;
    string imageNum = ossID.str() + ("/0") + ossCnt.str();
    putText(img, imageNum, Point(50,55), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255));
}
centroidInfo getToolInfoOnOneImage(const uint & ID, const uint & cntImg, Mat & imageRGB){
    IPParameters IPparams(ID);
    centroidInfo toolInfo;
    toolInfo.setSeqID(ID);
    toolInfo.setImgIndex(cntImg);
    fillConvexPoly(imageRGB, IPparams.getMaskCorners(), Scalar(0,0,0));
    //segment blue using bounds
    Mat imageBlue;
    inRange(imageRGB, IPparams.getMinColor(), IPparams.getMaxColor(), imageBlue);
    //morphology
    //imageBlue = MorphologyOperations(imageBlue, 3, 1, 5);
    //Get the center of the blob
    vector<Point> lContour = GetLargestContour(imageBlue, 20);
    toolInfo.setPos(-1,-1);
    if (lContour.size() > 0) {
        Moments mu;
        mu = moments( lContour, false ); // get the moment
        toolInfo.setPos( mu.m10/mu.m00 , mu.m01/mu.m00); // get the mass center
        //draw center in red and lcontour in green
        circle(imageRGB, toolInfo.getPos(), 3, Scalar(0,0,255), 1);
        for (uint i = 0; i < lContour.size(); i++) {
            circle(imageRGB, lContour.at(i), 1, Scalar(0,255,0)); // draw circle
        }
    }
    writePathOnImage(imageRGB, ID, cntImg);
    imshow( "CUR", imageRGB);
    imshow( "PROCIMAGE", imageBlue);
    waitKey(50);              // Wait for a keystroke in the window
    return(toolInfo);
}
vector <centroidInfo> trackTool(const Vec3i & annotatedSequences){ // Vec3i= 3d vector of int  items
    vector <centroidInfo> toolInfoHistory;
    const uint ID = annotatedSequences[0];
    ostringstream ossID, ossType;
    ossID << setw(3) << setfill('0') << ID;
    string fName = ("./images/tool/") + ossID.str() + string("/");
    const uint startCnt = annotatedSequences[1];
    const uint endCnt= annotatedSequences[2];
    Mat imageRGB;
    for (uint cntImg = startCnt; cntImg <= endCnt; cntImg++) {
        imageRGB = LoadImage(cntImg, true, fName);
        centroidInfo oneToolInfo = getToolInfoOnOneImage(ID, cntImg, imageRGB);
        toolInfoHistory.push_back(oneToolInfo);
    }
    return(toolInfoHistory);
}
vector <Vec3i> loadSubSets(const char * fName){
    vector <Vec3i> subSets;
    Vec3i thisSubSet;
    ifstream subSetsFile;
    subSetsFile.open(fName);
    uint numSubSets;
    if (subSetsFile.is_open()) {
        cout << "Open" << endl;
        subSetsFile >> numSubSets;
        for (uint i =0; i < numSubSets; i++) {
            subSetsFile >> thisSubSet[0];
            subSetsFile >> thisSubSet[1];
            subSetsFile >> thisSubSet[2];
            subSets.push_back(thisSubSet);
        }
    }
    subSetsFile.close();
    return(subSets);
}
int main() {
    namedWindow( "CUR", CV_WINDOW_AUTOSIZE );moveWindow("CUR", 0, 0);
    namedWindow( "PROCIMAGE", CV_WINDOW_AUTOSIZE );moveWindow("PROCIMAGE", 800, 0);



    ofstream myfile;
    myfile.open ("filename.txt");               
    myfile << "Writing this to a file.\n";
    myfile.close();
    //track tool position and save it to file
#define IMAGES_ON_DISK
#ifdef IMAGES_ON_DISK
    //save centroids to a file
    cout << "1" << endl;
    ofstream centroidFile;
    vector <Vec3i> annotatedSequences = loadSubSets("./images/blueBlob.txt"); // If the second argument is not specified, it is implied CV_LOAD_IMAGE_COLOR: loads the image in the BGR format
    cout << "2" << endl;
    cout << annotatedSequences.size() << endl;
    centroidFile.open("centroidFile.txt");
    for (uint i=0; i < annotatedSequences.size(); i++) {
        vector <centroidInfo> toolCentroidsRaw = trackTool(annotatedSequences[i]);
        //log to file raw position
        for (uint j = 0; j < toolCentroidsRaw.size(); j++) {
            centroidFile << toolCentroidsRaw[j];
            centroidFile << endl;
        }
    }
    centroidFile.close();
#else
    uint ID = 1;
    uint cnt = 1;
    while (cnt < 100) {
        Mat imageRGB = imread("./images/tool/007/rgb/00001.png");
        centroidInfo toolInfo = getToolInfoOnOneImage(ID, cnt, imageRGB);
        cnt++;
    }
#endif
    cout << "*********** MAIN END!! **********" << endl;
    return 0;
}
