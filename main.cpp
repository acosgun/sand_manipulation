#include "mainwindow.h"
#include <QApplication>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "/usr/include/opencv2/opencv.hpp"
#include "/usr/include/opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class centroidInfo {
public:
    centroidInfo(){
        setPos(-1,-1);
        setVel(0,0);
        setSeqID(0);
        setImgIndex(0);
        setVelChange(false);
        setVisible(false);
    }
    ~centroidInfo(){}
    centroidInfo operator=(const centroidInfo & other) {
        this->setPos(other.getPos());
        this->setVel(other.getVel());
        this->setSeqID(other.getSeqID());
        this->setImgIndex(other.getImgIndex());
        this->setVelChange(other.getVelChange());
        this->setVisible(other.getVisible());
        return *this;
    }
    Point2f getPos() const {return (pos);}
    Point2f getVel() const {return (vel);}
    int getSeqID() const {return (seqID);}
    int getImgIndex() const {return (imgIndex);}
    bool getVelChange() const {return(velChange);}
    bool getVisible() const {return(visible);}
    void setPos(const Point2f & p){pos = p;}
    void setVel(const Point2f& v){vel = v;}
    void setPos(const float & px, const float & py){pos.x = px; pos.y = py;}
    void setVel(const float & vx, const float & vy){vel.x = vx; vel.y = vy;}
    void setSeqID(const int & id){seqID = id;}
    void setImgIndex(const int & index){imgIndex = index;}
    void setVelChange(const bool & vc){velChange = vc;}
    void setVisible(const bool & vis){visible = vis;}
private:
    Point2f pos;
    Point2f vel;
    int seqID;
    int imgIndex;
    bool velChange;
    bool visible;
};
class sampledContour: public vector<Point> {
public:
    sampledContour(){
        this->resize(sampleCnt);
    }
    ~sampledContour(){}
private:
    static const uint sampleCnt = 10;
};
sampledContour sortContourSamplesByIncreasingY(const sampledContour & toSort) {
    sampledContour sorted;
    const int num = toSort.size();
    vector <bool> done;
    done.resize(num);
    sorted.resize(num);
    for (int i = 0; i < num; i++) done[i] = false;
    int minY = 1000;
    int minIndex = -1;
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < num; j++) {
            if ((toSort[j].y < minY) && (!done[j])) {
                minY = toSort[j].y;
                minIndex = j;
            }
         }
        sorted[i] = toSort[minIndex];
        done[minIndex] = true;
        minY = 1000;
    }
    return sorted;
}
float getDistanceBetweenContours(const sampledContour & cont1, const sampledContour & cont2){
    float contDist_ = 0;
    for (uint i = 0; i < cont1.size(); i++) {
        contDist_ += norm(cont1[i]-cont2[i]);
    }
    contDist_ /= cont1.size();
    return(contDist_);
}
class imageData {
public:
    imageData(){}
    ~imageData(){}
    centroidInfo centInfo;
    sampledContour sandContour;
};
class IPParameters{
public:
    IPParameters(){}
    IPParameters(const uint & ID) {
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
private:
    vector<Point> maskCorners;
    bool maskHand;
};
istream& operator>>(istream &cInput, centroidInfo &c){
    int id, index, vc, vis;
    float px, py, vx, vy;
    cInput>> id;
    cInput>> index;
    cInput>> px;
    cInput>> py;
    cInput>> vx;
    cInput>> vy;
    cInput>> vc;
    cInput>> vis;
    c.setImgIndex(index);
    c.setSeqID(id);
    c.setPos(px, py);
    c.setVel(vx, vy);
    c.setVelChange((bool)(vc));
    c.setVisible((bool)(vis));
    return(cInput);
}
ostream& operator<<(ostream &cOutput, const centroidInfo &c){
    cOutput << c.getSeqID() << "\t" << c.getImgIndex() << "\t"
            << c.getPos().x << "\t" << c.getPos().y << "\t"
            << c.getVel().x << "\t" << c.getVel().y << "\t"
            << c.getVelChange() << "\t" << c.getVisible()  << "\t";
    return(cOutput);
}
//returns all contours enclosing area larger than min_pixel
vector< vector<Point> > GetBigContours(const Mat & srcImg, const float & min_pixel_){
    vector< vector<Point> > bigContours;
    bigContours.clear();
    if(srcImg.rows!=0 && srcImg.cols!=0){
        //finding contours
        vector<Vec4i> hierarchy;
        Mat to_contour = srcImg.clone();
        vector< vector<Point> > contours;
        findContours(to_contour, contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE ); // Find the contours in the image
        //find largest
        for(unsigned int i = 0; i< contours.size(); i++ ) {// iterate through each contour.
            if(contourArea(contours[i],false) > min_pixel_){//the blob can be considered if it contains enough pixels
                bigContours.push_back(contours[i]);
            }
        }
    }
    if(bigContours.size() == 0){//at least one contour needed !
        cout << "no big contour found " << endl;
    }
    return(bigContours);
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
int getLargestContourIndex(const vector< vector<Point> > & bigContours_) {
    float largest_area = 0;
    int largest_area_index = -1;
    for(unsigned int i = 0; i< bigContours_.size(); i++ ) {// iterate through each contour.
        if(contourArea(bigContours_[i],false)>largest_area){
            largest_area = contourArea(bigContours_[i],false);
            largest_area_index = i;
        }
    }
    return (largest_area_index);
}
Rect MergeBoundingBoxes(const vector< vector<Point> > & bigContours, const float & max_distance_for_merge){
    Rect largest_bounding_rect;
    int largest_area_index_ = getLargestContourIndex(bigContours);
    vector<Rect> areas_found;
    for(unsigned int i = 0; i< bigContours.size(); ++i ) {// iterate through each area.
        areas_found.push_back(boundingRect(bigContours[i]));
    }
    largest_bounding_rect = areas_found[largest_area_index_]; // Find the bounding rectangle for biggest contour
    for(unsigned int i = 0; i< bigContours.size(); ++i ) {// iterate through each area.
        if ((int)i != largest_area_index_){
            //creating the rectangle englobing this area
            Rect curr_bounding_rect = areas_found[i];
            vector<Point> c;//centers
            c.push_back(Point(curr_bounding_rect.x+curr_bounding_rect.width/2, curr_bounding_rect.y+curr_bounding_rect.height/2));
            c.push_back(Point(largest_bounding_rect.x+largest_bounding_rect.width/2, largest_bounding_rect.y+largest_bounding_rect.height/2));
            float d = sqrt((c[0].x-c[1].x)*(c[0].x-c[1].x)+(c[0].y-c[1].y)*(c[0].y-c[1].y));
            if (d/sqrt(largest_bounding_rect.width*largest_bounding_rect.width+largest_bounding_rect.height*largest_bounding_rect.height) < max_distance_for_merge){
                Point largest_bounding_rectBR(largest_bounding_rect.x+largest_bounding_rect.width, largest_bounding_rect.y+largest_bounding_rect.height);
                Point this_bounding_rectBR(curr_bounding_rect.x+curr_bounding_rect.width, curr_bounding_rect.y+curr_bounding_rect.height);
                if (largest_bounding_rect.x >  curr_bounding_rect.x) largest_bounding_rect.x = curr_bounding_rect.x;
                if (largest_bounding_rect.y >  curr_bounding_rect.y) largest_bounding_rect.y = curr_bounding_rect.y;
                if (largest_bounding_rectBR.x <  this_bounding_rectBR.x) largest_bounding_rectBR.x = this_bounding_rectBR.x;
                if (largest_bounding_rectBR.y <  this_bounding_rectBR.y) largest_bounding_rectBR.y = this_bounding_rectBR.y;
                largest_bounding_rect.width=largest_bounding_rectBR.x-largest_bounding_rect.x;
                largest_bounding_rect.height=largest_bounding_rectBR.y-largest_bounding_rect.y;
            }
        }
    }
    return(largest_bounding_rect);
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
vector <Point> GetDiffContour(const Rect & roi, const Mat & img){
    vector<Point> theContour;
    Mat desROI = img(roi);
    // convert image to hsv
    Mat hsv_image;
    cvtColor(desROI, hsv_image, CV_BGR2HSV);
    //segment according to levels
    Mat segImg;
    Scalar minHSV;
    Scalar maxHSV;
    minHSV = Scalar(0,0,50);
    maxHSV = Scalar(255,50,255);
    inRange(hsv_image, minHSV, maxHSV, desROI);
    //if (desROI.cols!=0 && desROI.rows!=0) namedWindow("ROI", CV_WINDOW_AUTOSIZE );moveWindow("ROI", 1600, 600); imshow("ROI", desROI); waitKey(10);
    vector< vector<Point> > bigContours = GetBigContours(desROI, 100);//TODO PARAMS
    if (bigContours.size()==0) {
        cout << "in Get Diff Contour no bigContours "<<endl;
    } else {
        int d = 2;//do not account for border pixels
        //for(unsigned int i = 0; i< bigContours.size(); i++ ) {
        unsigned int i = 0;
        for(unsigned int j = 0; j< bigContours.at(i).size(); j++ ) {
            Point pt = bigContours.at(i).at(j);
            if ((pt.x > d) && (pt.y > d) && (pt.x < desROI.size().width-d) && (pt.y < desROI.size().height-d)) {
                theContour.push_back(pt+roi.tl());
            }
        }
        //}
    }
    return(theContour);
}
sampledContour SampleContour(const uint & newSize, const vector <Point> & oldVec) {
    sampledContour newVec;
    Point ptInter;
    double scale = ((double)(oldVec.size())) / ((double)(newSize));
    double integPart;
    for (uint i = 0; i < newSize; i++) {
        double floatPart = modf(scale*i, &integPart);
        if (integPart + 1 >= oldVec.size()) integPart = oldVec.size()-2;
        ptInter = oldVec.at((uint)(integPart)) + floatPart*(oldVec.at((uint)(integPart+1))-oldVec.at((uint)(integPart)));
        newVec[i] = ptInter;
    }
    return newVec;
}
Mat LoadImage(const int & cntImg, const bool & rgb, const string & fName){//TODO provide just folder name as argument
    Mat image;
    ostringstream ossCnt, ossType;
    ossCnt << setw(4) << setfill('0') << cntImg;
    if (rgb) ossType.str("rgb");
    else ossType.str("depth");
    string imageName = fName + ossType.str() + ("/0") + ossCnt.str() + string(".png");
    cout << "==== IMAGE " << imageName << " ====" << endl;
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
Mat trackHandAndArm(const Mat & Img){
    Mat hImg;
    Mat htmpImg;
    cvtColor(Img, htmpImg, CV_BGR2HSV);
    //segment according to levels
    Scalar minHSV = Scalar(0,0,0);
    Scalar maxHSV = Scalar(255,255,50);
    inRange(htmpImg, minHSV, maxHSV, htmpImg);
    vector < vector<Point> > bigCont = GetBigContours(htmpImg, 500);
    Rect handROI;
    if (bigCont.size()==0) {
        cout << "in trackHandAndArm no contours found " << handROI << endl;
    } else {
        handROI = MergeBoundingBoxes(bigCont, 0.5);
    }
    vector< vector<Point> >  handContours;
    Mat hROI = htmpImg(handROI);
    hImg = Mat::zeros(htmpImg.rows, htmpImg.cols, htmpImg.type());
    findContours(hROI, handContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, handROI.tl());
    drawContours(hImg, handContours, -1, Scalar(255,255,255), CV_FILLED);
    //draw roi in hand image
    //rectangle(hImg, handROI, Scalar(255,255,255));
    return(hImg);
}
void processImageSequencePush(const Vec3i & pushSeq, const vector <centroidInfo> & seqCentroidInfo, ofstream & lFile, const bool & showImages){
    const uint ID = pushSeq[0];
    const uint startCnt = pushSeq[1];
    const uint endCnt= pushSeq[2];

    //load start and end image
    ostringstream ossID;
    ossID << setw(3) << setfill('0') << ID;
    string folderName = ("../images/tool/") + ossID.str() + string("/");
    Mat startImg = LoadImage(startCnt, true, folderName);
    Mat endImg = LoadImage(endCnt, true, folderName);
    //mask part of the images
    IPParameters IPparams(ID);
    fillConvexPoly(startImg, IPparams.getMaskCorners(), Scalar(0,0,0));
    fillConvexPoly(endImg, IPparams.getMaskCorners(), Scalar(0,0,0));
    //subtract start and end image
    Mat grayErrImg;
    subtract(startImg, endImg, grayErrImg);
    cvtColor( grayErrImg, grayErrImg, CV_BGR2GRAY );
    //Mask hand if needed
    if (IPparams.getMaskHand()) {
        //get black parts (hand and mask) in start and end images
        Mat handStartImg = trackHandAndArm(startImg);
        Mat handEndImg = trackHandAndArm(endImg);
        Mat handBothImg;
        max(handStartImg, handEndImg, handBothImg);
        threshold(handBothImg, handBothImg, 127, 255, THRESH_BINARY_INV);
        if (showImages) { namedWindow("HAND", CV_WINDOW_AUTOSIZE );moveWindow("HAND", 0, 600);imshow("HAND", handStartImg);}
        if (showImages) { namedWindow("HAND2", CV_WINDOW_AUTOSIZE );moveWindow("HAND2", 700, 600);imshow("HAND2", handBothImg);}
        if (showImages) { namedWindow("HAND3", CV_WINDOW_AUTOSIZE );moveWindow("HAND3", 1400, 600);imshow("HAND3", handEndImg);}
        min(grayErrImg, handBothImg, grayErrImg);
    }
    //from difference between start and end image get Region Of Interest
    static int binThre = 50;//TODO PARAMS
    threshold(grayErrImg, grayErrImg, binThre, 255, THRESH_BINARY);
    //namedWindow("THRESH", CV_WINDOW_AUTOSIZE );moveWindow("THRESH", 0, 600);imshow("THRESH", grayErrImg);
    grayErrImg = MorphologyOperations(grayErrImg, 3, 1, 10);
    //namedWindow("MORPH", CV_WINDOW_AUTOSIZE );moveWindow("MORPH", 1400, 600);imshow("MORPH", grayErrImg);
    vector < vector<Point> > bigCont = GetBigContours(grayErrImg, 500);
    Rect StartToEndDiffROIPush;
    if (bigCont.size()==0) {
        cout << "No contours in diff image so Region Of Interest is " << StartToEndDiffROIPush << endl;
    } else {
        StartToEndDiffROIPush = MergeBoundingBoxes(bigCont, 0.8);
    }
    //draw roi in end and in start image
    rectangle(endImg, StartToEndDiffROIPush, Scalar(255,0,0));
    rectangle(startImg, StartToEndDiffROIPush, Scalar(255,0,0));
    //draw in green the end contour in end image
    vector<Point>  endSandContour = GetDiffContour(StartToEndDiffROIPush,endImg);
    for (uint i = 0; i < endSandContour.size(); i++) {
        circle(endImg, endSandContour.at(i), 1, Scalar(0,255,0));
    }
    //draw in red the start contour in start image
    vector<Point> startSandContour = GetDiffContour(StartToEndDiffROIPush, startImg);
    for (uint i = 0; i < startSandContour.size(); i++) {
        circle(startImg, startSandContour.at(i), 1, Scalar(0,0,255));
    }
    if (showImages) {
        namedWindow("START", CV_WINDOW_AUTOSIZE ); moveWindow("START", 0, 0); imshow("START", startImg);
        namedWindow("END", CV_WINDOW_AUTOSIZE ); moveWindow("END", 1400, 0); imshow( "END", endImg);
        waitKey(10);
    }
    //LOOP
    Mat curImg;
    sampledContour prevSandContourSampled;
    for (uint cntImg = startCnt; cntImg <= endCnt; cntImg++) {
        curImg = LoadImage(cntImg, true, folderName);
        fillConvexPoly(curImg, IPparams.getMaskCorners(), Scalar(0,0,0));
        if (endSandContour.size() == 0) {
            cout << "end SandContour does not exist!" << endl;
            //waitKey(10);
        } else {
            vector<Point> curSandContour = GetDiffContour(StartToEndDiffROIPush, curImg);
            sampledContour curSandContourSampled;
            if (curSandContour.size() == 0) {
                cout << "cur SandContour does not exist!" << endl;
                for (uint j = 0; j < curSandContourSampled.size(); j++) {
                    curSandContourSampled.at(j).x=-1;
                    curSandContourSampled.at(j).y=-1;
                }
             } else {
                //sample to make same size
                curSandContourSampled = SampleContour(curSandContourSampled.size(),curSandContour);
                for (uint j = 0; j < curSandContourSampled.size(); j++) {
                    circle(curImg, curSandContourSampled.at(j), 2, Scalar(255,0,0));
                }
            }
            //save to file
            centroidInfo thisCentroidInfo = seqCentroidInfo[cntImg-startCnt];
            if ((cntImg == startCnt) || (cntImg == endCnt)) {
                thisCentroidInfo.setVelChange(true);
            }
            lFile << thisCentroidInfo;
            for (uint j = 0; j < curSandContourSampled.size(); j++) {
                lFile << curSandContourSampled.at(j).x << "\t" << curSandContourSampled.at(j).y << "\t";
            }
            lFile << endl;
            if (showImages) {
                //draw features
                namedWindow( "CUR", CV_WINDOW_AUTOSIZE );moveWindow("CUR", 700, 0);
                rectangle(curImg, StartToEndDiffROIPush, Scalar(255,0,0));
                Point2f centroidPosition = seqCentroidInfo[cntImg-startCnt].getPos();
                circle(curImg, centroidPosition, 4, Scalar(255,255,255));
                writePathOnImage(curImg, ID, cntImg);
                imshow( "CUR", curImg);
                waitKey(10);
            }
            prevSandContourSampled = curSandContourSampled;

        }
    }
}
//returns a vector of centroid info over a sequence of images and show images if flag is true
vector <centroidInfo> trackToolCentroid(const Vec3i & annotatedSequences, const bool & showImages){
    vector <centroidInfo> toolInfo;
    centroidInfo oneToolInfo;
    const uint ID = annotatedSequences[0];
    oneToolInfo.setSeqID(ID);
    ostringstream ossID, ossType;
    ossID << setw(3) << setfill('0') << ID;
    string fName = ("../images/tool/") + ossID.str() + string("/");
    const uint startCnt = annotatedSequences[1];
    const uint endCnt= annotatedSequences[2];
    Mat imageRGB;
    Mat imageBlue;
    if (showImages) {
        namedWindow( "CUR", CV_WINDOW_AUTOSIZE );moveWindow("CUR", 0, 0);
        namedWindow( "PROCIMAGE", CV_WINDOW_AUTOSIZE );moveWindow("PROCIMAGE", 800, 0);
    }
    //blue bounds
    Scalar min_Color;
    Scalar max_Color;
    min_Color[0] = 0; min_Color[1] = 70;    min_Color[2] = 0;   //TODO PARAMS
    max_Color[0] = 255; max_Color[1] = 200;  max_Color[2] = 20;
    IPParameters IPparams(ID);
    for (uint cntImg = startCnt; cntImg <= endCnt; cntImg++) {
        oneToolInfo.setImgIndex(cntImg);
        imageRGB = LoadImage(cntImg, true, fName);
        fillConvexPoly(imageRGB, IPparams.getMaskCorners(), Scalar(0,0,0));
        //segment blue using bounds
        inRange(imageRGB, min_Color, max_Color, imageBlue);
        //morphology
        //imageBlue = MorphologyOperations(imageBlue, 3, 1, 5);
        //Get the center of the blob
        vector<Point> lContour = GetLargestContour(imageBlue, 20);
        oneToolInfo.setPos(-1,-1);
        if (lContour.size() > 0) {
            oneToolInfo.setVisible(true);
            Moments mu;
            mu = moments( lContour, false );
            oneToolInfo.setPos( mu.m10/mu.m00 , mu.m01/mu.m00);
            //draw center in red and lcontour in green
            circle(imageRGB, oneToolInfo.getPos(), 3, Scalar(0,0,255), 1);
            for (uint i = 0; i < lContour.size(); i++) {
                circle(imageRGB, lContour.at(i), 1, Scalar(0,255,0));
            }
        } else {
            oneToolInfo.setVisible(false);
        }
        toolInfo.push_back(oneToolInfo);
        if (showImages) {
            writePathOnImage(imageRGB, ID, cntImg);
            imshow( "CUR", imageRGB);
            imshow( "PROCIMAGE", imageBlue);
            waitKey(10);
        }
    }
    return(toolInfo);
}
//saves sequences as a vector of triplets of integers [ID startImg endImg]
vector <Vec3i> loadSubSets(const char * fName){
    vector <Vec3i> subSets;
    Vec3i thisSubSet;
    ifstream subSetsFile;
    subSetsFile.open(fName);
    uint numSubSets;
    if (subSetsFile.is_open()) {
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
vector <centroidInfo> fillTrackingHoles(const vector <centroidInfo> & oldCentroids){
    vector <centroidInfo> newCentroids;
    newCentroids = oldCentroids;
    uint j = 0;
    uint n = 1;
    while (j < oldCentroids.size()) {
        if (oldCentroids[j].getPos().x >= 0.) {
            j++;
            n = j+1;
        } else {
            if (oldCentroids[n].getPos().x < 0) {
                n++;
            } else {
                if (j==0) cout << "impossible to fill holes since tool not visible in first image " << endl;
                float stepX = (oldCentroids[n].getPos().x-oldCentroids[j-1].getPos().x)/(n-j+1);
                float stepY = (oldCentroids[n].getPos().y-oldCentroids[j-1].getPos().y)/(n-j+1);
                for (uint m = j; m < n; m++) {
                    newCentroids[m].setPos(oldCentroids[j-1].getPos().x +  (m-j+1)*stepX,
                            oldCentroids[j-1].getPos().y +  (m-j+1)*stepY);
                }
                j=n;
            }
        }
    }
    return(newCentroids);
}
vector <centroidInfo> applyFilter(const vector <centroidInfo> & oldCentroids){
    vector <centroidInfo> newCentroids;
    newCentroids = oldCentroids;
    newCentroids[0].setPos((2*oldCentroids[0].getPos().x + oldCentroids[1].getPos().x)/3,
            (2*oldCentroids[0].getPos().y + oldCentroids[1].getPos().y)/3);
    uint j = 1;
    while (j < oldCentroids.size()-1) {
        newCentroids[j].setPos((oldCentroids[j-1].getPos().x + 2*oldCentroids[j].getPos().x + oldCentroids[j+1].getPos().x)/4,
                (oldCentroids[j-1].getPos().y + 2*oldCentroids[j].getPos().y + oldCentroids[j+1].getPos().y)/4);
        j++;
    }
    newCentroids[j].setPos((oldCentroids[j-1].getPos().x + 2*oldCentroids[j].getPos().x)/3,
            (oldCentroids[j-1].getPos().y + 2*oldCentroids[j].getPos().y)/3);
    return(newCentroids);
}
vector <centroidInfo> setVelocityAndChange(const vector <centroidInfo> & oldC){
    vector <centroidInfo> newC;
    //below this a change is considered
    static const float minVel = .5;//TODO PARAMS
    newC = oldC;
    for (uint j = 0; j < newC.size()-1; j++) {
        newC[j].setVel(oldC[j+1].getPos().x - oldC[j].getPos().x, oldC[j+1].getPos().y - oldC[j].getPos().y);
    }
    newC[newC.size()-1].setVel(newC[newC.size()-2].getVel());
    newC[0].setVelChange(true);
    for (uint j = 1; j < newC.size()-1; j++) {
        bool change = (newC[j].getVel().x*newC[j-1].getVel().x + newC[j].getVel().y*newC[j-1].getVel().y < 0) ||
                (sqrt((newC[j-1].getVel().x)*(newC[j-1].getVel().x) + (newC[j-1].getVel().y)*(newC[j-1].getVel().y) < minVel));
        newC[j].setVelChange(change);
    }
    newC[newC.size()-1].setVelChange(true);
    return(newC);
}
void getAndSaveCentroidsAndMotPrimSequences(const vector <Vec3i> annotatedSequences, const bool & showImages){
    vector<Vec3i> motPrimSeqPush;
    vector<uint> imgWithToolNotVisible;
    vector<uint> IDWithToolNotVisible;
    vector<Point2f> toolTapPos;

    //save centroids to a file
    ofstream centroidFileRaw;
    centroidFileRaw.open("centroidFileRaw.txt");
    ofstream centroidFileProcessed;
    centroidFileProcessed.open("centroidFileProcessed.txt");
    //loop all subsets
    for (uint i=0; i < annotatedSequences.size(); i++) {
        //track the tool coordinates on this annotated sequence
        vector <centroidInfo> toolCentroidsRaw = trackToolCentroid(annotatedSequences[i], showImages);
        //log to file raw position
        for (uint j = 0; j < toolCentroidsRaw.size(); j++) {
            centroidFileRaw << toolCentroidsRaw[j];
            centroidFileRaw << endl;
        }
        //process centroid position to detect changes
        vector <centroidInfo> toolCentroidsFilled = fillTrackingHoles(toolCentroidsRaw);
        vector <centroidInfo> toolCentroidsFiltered = applyFilter(toolCentroidsFilled);
        vector <centroidInfo> toolCentroidsFinal = setVelocityAndChange(toolCentroidsFiltered);
        //log to file processed position
        for (uint j = 0; j < toolCentroidsFinal.size(); j++) {
            centroidFileProcessed << toolCentroidsFinal[j];
            centroidFileProcessed << endl;
        }

        //generate motPrimSequences for tapping
        uint thisImgWithToolNotVisible = 0;
        uint cntImgNotVis = 0;
        Point2f thisCentroidPos(0,0);
        uint cntImgVis = 0;
        float velThresh = 0.7;
        for (uint j = 0; j < toolCentroidsFinal.size(); j++) {
            if (!toolCentroidsFinal[j].getVisible()) {
                if (cntImgVis > 0) {
                    cout << "there were " << cntImgVis << " slow points in this series " << endl;
                    thisCentroidPos.x = thisCentroidPos.x / cntImgVis;
                    thisCentroidPos.y = thisCentroidPos.y / cntImgVis;
                    toolTapPos.push_back(thisCentroidPos);
                    thisCentroidPos.x = 0;
                    thisCentroidPos.y = 0;
                    cntImgVis = 0;
                }
                thisImgWithToolNotVisible += toolCentroidsFinal[j].getImgIndex();
                cntImgNotVis++;
            } else {
                if (cntImgNotVis > 0) {
                    //get average index of consecutive images with invisible tool and store it in the vector
                    thisImgWithToolNotVisible /= cntImgNotVis;
                    imgWithToolNotVisible.push_back(thisImgWithToolNotVisible);
                    IDWithToolNotVisible.push_back(toolCentroidsFinal[j].getSeqID());
                    thisImgWithToolNotVisible = 0;
                    cntImgNotVis = 0;
                }
                //update the sum of slow points
                if (norm(toolCentroidsFinal[j].getVel()) < velThresh){
                    thisCentroidPos.x = thisCentroidPos.x + (toolCentroidsFinal[j].getPos()).x;
                    thisCentroidPos.y = thisCentroidPos.y + (toolCentroidsFinal[j].getPos()).y;
                    cntImgVis++;
                }
            }
        }
        //generate motPrimSequences for pushing
        cout << "keyframes start " << endl;
        vector<uint> imgWithVelChange;
        imgWithVelChange.clear();
        for (uint j = 0; j < toolCentroidsFinal.size(); j++) {
            if (toolCentroidsFinal[j].getVelChange()) {
                imgWithVelChange.push_back(toolCentroidsFinal[j].getImgIndex());
            }
        }
        cout << "keyframes done " << endl;
        Vec3i thisMotPrim;
        uint toChange = motPrimSeqPush.size();
        for (uint j = 0; j < imgWithVelChange.size()-1; j++) {
            thisMotPrim[0] = annotatedSequences[i][0];
            thisMotPrim[1] = imgWithVelChange[j]+1;
            thisMotPrim[2] = imgWithVelChange[j+1];
            motPrimSeqPush.push_back(thisMotPrim);
        }
        motPrimSeqPush[toChange][1] = annotatedSequences[i][1];
    }
    centroidFileRaw.close();
    centroidFileProcessed << "\n";
    centroidFileProcessed.close();

    //data files for tapping
    ofstream U_tap_images,
            Y_tap;
    U_tap_images.open("U_tap_images.txt");//[userID img_ini img_fin]
    for (uint i = 0; i < imgWithToolNotVisible.size()-1; i++) {
        U_tap_images << IDWithToolNotVisible[i] << "\t" << imgWithToolNotVisible[i] << "\t" << imgWithToolNotVisible[i+1] << endl;
    }
    U_tap_images.close();
    Y_tap.open("Y_tap.txt");//[centX centY]
    for (uint i = 1; i < toolTapPos.size(); i++) {
        Y_tap << toolTapPos[i].x << "\t" << toolTapPos[i].y << endl;
    }
    Y_tap.close();
    cout << " imgWithToolNotVisible size is " << imgWithToolNotVisible.size() << endl;
    for (uint i = 0; i < imgWithToolNotVisible.size(); i++)
        cout << imgWithToolNotVisible[i] << endl;
    cout << " toolTapPos " << toolTapPos << endl;

    //mot prim file for pushing
    ofstream motPrimFilePush;
    motPrimFilePush.open("motPrimPush.txt");
    //Min number of images in a motion primitive sequence
    static const int minImgInSeq = 10;//TODO PARAMS
    for (uint j = 0; j < motPrimSeqPush.size(); j++){
        if ((motPrimSeqPush[j][2] - motPrimSeqPush[j][1]) > minImgInSeq) {
            motPrimFilePush << motPrimSeqPush[j][0] << "\t" << motPrimSeqPush[j][1] << "\t" << motPrimSeqPush[j][2] <<  endl;
        }
    }
    motPrimFilePush.close();
}
vector <Vec3i> loadMotPrimitives(const char * motPrimFName){
    vector <Vec3i> motPrimSequence;
    Vec3i thisMotPrim;
    ifstream motPrimFileIn;
    motPrimFileIn.open(motPrimFName);
    while (!motPrimFileIn.eof()) {
        motPrimFileIn >> thisMotPrim[0];
        motPrimFileIn >> thisMotPrim[1];
        motPrimFileIn >> thisMotPrim[2];
        cout << "thisMotPrim is " << thisMotPrim<< endl;

        motPrimSequence.push_back(thisMotPrim);
    }
    motPrimSequence.resize(motPrimSequence.size()-1);//TODO do this nicely with eof
    motPrimFileIn.close();
    cout << "motPrimSequence.size() is " << motPrimSequence.size()<< endl;
    return(motPrimSequence);
}
vector <centroidInfo> loadAllCentroidInfo(const char * cenProcFName){
    vector <centroidInfo> allCentroidInfo;
    centroidInfo thisCentroidInfo;
    ifstream cenProcFileIn;
    cenProcFileIn.open(cenProcFName);
    while (!cenProcFileIn.eof()) {
        cenProcFileIn >> thisCentroidInfo;
        allCentroidInfo.push_back(thisCentroidInfo);
    }
    allCentroidInfo.resize(allCentroidInfo.size()-1);//TODO do this nicely with eof
    cout << "allCentroidInfo.size() is " << allCentroidInfo.size()<< endl;
    cenProcFileIn.close();
    return(allCentroidInfo);
}
vector <centroidInfo> getCentroidInfoForSequence(const Vec3i & mPrimSeq, const vector <centroidInfo> & allCentroidInfo){
    vector <centroidInfo> cInfo;
    centroidInfo thisCInfo;
    for (uint i = 0; i<allCentroidInfo.size(); i++){
        thisCInfo = allCentroidInfo[i];
        if ((thisCInfo.getSeqID()==mPrimSeq[0]) && (thisCInfo.getImgIndex()>=mPrimSeq[1]) && (thisCInfo.getImgIndex()<=mPrimSeq[2])) {
            cInfo.push_back(thisCInfo);
        }
    }
    return(cInfo);
}
void generateDataForLearningPush(const char * seqFileName, const char * imagesFileName, const char * contoursFileName, const char * actionsFileName){
    ifstream seqFile;
    seqFile.open(seqFileName);
    vector < imageData > sandmanPush;
    while (!seqFile.eof()) {
        imageData thisImgData;
        seqFile >> thisImgData.centInfo;
        for (uint i = 0; i < thisImgData.sandContour.size(); i++) {
            seqFile >> thisImgData.sandContour[i].x;
            seqFile >> thisImgData.sandContour[i].y;
        }
        sandmanPush.push_back(thisImgData);
    }
    sandmanPush.resize(sandmanPush.size()-1);
    seqFile.close();
    cout << "sandman size is " << sandmanPush.size() << endl;
    uint j = 0;
    uint seqNb = 0;
    vector < vector < imageData > > dataBySeq;
    //table containing for each sequence: ID, startImg, endImg
    vector <Vec3i> seq;
    seq.resize(sandmanPush.size());
    dataBySeq.resize(sandmanPush.size());
    for(uint img = 0; img < sandmanPush.size(); img++) {
        dataBySeq[seqNb].push_back(sandmanPush[img]);
        if (sandmanPush[img].centInfo.getVelChange()) {
            if ((j % 2) == 0) {
                seq[seqNb][0] = sandmanPush[img].centInfo.getSeqID();
                seq[seqNb][1] = sandmanPush[img].centInfo.getImgIndex();
                cout << "seq " << seqNb << "\t" << seq[seqNb][0] << "\t" << seq[seqNb][1];
            } else {
                seq[seqNb][2] = sandmanPush[img].centInfo.getImgIndex();
                cout << " -- " << seq[seqNb][2] << endl;
                seqNb++;
            }
            j++;
        }
    }
    seq.resize(seqNb);
    dataBySeq.resize(seqNb);
    cout << "\nseq size is " << seq.size() << endl;
    ofstream imagesFile;
    imagesFile.open(imagesFileName);
    ofstream contoursFile;
    contoursFile.open(contoursFileName);
    ofstream actionsFile;
    actionsFile.open(actionsFileName);
    int cnt = 0;
    int ncnt = 0;
    for (uint nSeq = 0; nSeq < seq.size(); nSeq++) {
        uint seqLength = seq[nSeq][2] - seq[nSeq][1] + 1;
        for (uint f = 0; f < seqLength - 1; f++) {
            for (uint s = f+1; s < seqLength; s++) {
                Point2f toolF = dataBySeq[nSeq][f].centInfo.getPos();
                Point2f toolS = dataBySeq[nSeq][s].centInfo.getPos();
                sampledContour contF = sortContourSamplesByIncreasingY(dataBySeq[nSeq][f].sandContour);
                sampledContour contS = sortContourSamplesByIncreasingY(dataBySeq[nSeq][s].sandContour);
                float toolDist = norm(toolF-toolS);
                float contDist = getDistanceBetweenContours(contF,contS);
                //bool leftBound = ((toolS.x - toolF.x) < 0); //TODO assumes left bound motion - generalize
                bool leftBound = true;
                if ((toolDist > 20) && (toolDist < 200) && (contDist > 30) && (contF[0].x > -1) && (contS[0].x > -1) && (leftBound)) {
                    imagesFile << dataBySeq[nSeq][f].centInfo.getSeqID() << "\t" << dataBySeq[nSeq][f].centInfo.getImgIndex() << "\t" << dataBySeq[nSeq][s].centInfo.getImgIndex() << "\n";
                    for (uint i = 0; i < contF.size(); i++) {
                        contoursFile << contF[i].x << "\t" << contF[i].y << "\t";
                    }
                    for (uint i = 0; i < contS.size(); i++) {
                        contoursFile << contS[i].x << "\t" << contS[i].y << "\t";
                    }
                    contoursFile  << endl;
                    actionsFile << (uint) toolF.x << "\t" << (uint) toolF.y << "\t" << (uint) toolS.x << "\t" << (uint) toolS.y << "\n";
                    cnt++;
                } else {
                    ncnt++;
                }
            }
        }
    }
    cout << "cnt " << cnt << " ncnt " << ncnt << endl;
    imagesFile.close();
    contoursFile.close();
    actionsFile.close();
}
void visualizeDataForLearningPush(const char * imagesFileName, const char * contoursFileName, const char * actionsFileName){
    //U_push_images: [userID img_ini img_fin]
    ifstream imagesFile;
    imagesFile.open(imagesFileName);
    vector <Vec3i> imageTriplets;
    Vec3i thisTriplet;
    while (!imagesFile.eof()) {
        imagesFile >> thisTriplet[0];
        imagesFile >> thisTriplet[1];
        imagesFile >> thisTriplet[2];
        imageTriplets.push_back(thisTriplet);
    }
    imagesFile.close();
    cout << "imageTriplets.size() is " << imageTriplets.size()<< endl;
    //U_push_contours: [X_ini,1 Y_ini,1 ... X_ini,n Y_ini,n X_fin,1 Y_fin,1 ... X_fin,n Y_fin,n]
    ifstream contoursFile;
    contoursFile.open(contoursFileName);
    vector < sampledContour > inContours;
    vector < sampledContour > finContours;
    while (!contoursFile.eof()) {
        sampledContour thisInContour;
        sampledContour thisFinContour;
        for (uint i = 0; i < thisInContour.size(); i++) {
            contoursFile >> thisInContour[i].x;
            contoursFile >> thisInContour[i].y;
        }
        inContours.push_back(thisInContour);
        for (uint i = 0; i < thisFinContour.size(); i++) {
            contoursFile >> thisFinContour[i].x;
            contoursFile >> thisFinContour[i].y;
        }
        finContours.push_back(thisFinContour);
    }
    contoursFile.close();
    cout << "inContours.size() is " << inContours.size()<< endl;
    cout << "finContours.size() is " << finContours.size()<< endl;
    //Y_push: [centX_ini centY_ini centX_fin centY_fin]
    ifstream actionsFile;
    actionsFile.open(actionsFileName);
    vector <Point> inAction;
    vector <Point> finAction;
    Point thisInAction;
    Point thisFinAction;
    while (!actionsFile.eof()) {
        actionsFile >> thisInAction.x;
        actionsFile >> thisInAction.y;
        actionsFile >> thisFinAction.x;
        actionsFile >> thisFinAction.y;
        inAction.push_back(thisInAction);
        finAction.push_back(thisFinAction);
    }
    actionsFile.close();
    cout << "inAction.size() is " << inAction.size() << endl;
    cout << "finAction.size() is " << finAction.size() << endl;
    //draw on img_ini points and contours and waitkey
    ofstream contoursFileManual;
    contoursFileManual.open("U_push_contours_V2.txt");
    ofstream actionFileManual;
    actionFileManual.open("Y_push_V2.txt");
    for (uint i = 0; i < inAction.size() - 1; i++) {
        //calculate average point of the two contours
        Point avgInContour;
        Point avgFinContour;
        for (uint j = 0; j < inContours[i].size(); j++) {
            avgInContour += inContours[i][j];
            avgFinContour += finContours[i][j];
        }
        avgInContour.x /= inContours[i].size();
        avgInContour.y /= inContours[i].size();
        avgFinContour.x /= finContours[i].size();
        avgFinContour.y /= finContours[i].size();
        //draw everything on two images
        cout << "\n====== drawing data " << i << " ---> " << imageTriplets[i][0] << " " << imageTriplets[i][1] << " " << imageTriplets[i][2] << endl;
        ostringstream ossID;
        ossID << setw(3) << setfill('0') << imageTriplets[i][0];
        string folderName = ("../images/tool/") + ossID.str() + string("/");
        Mat iniImg = LoadImage(imageTriplets[i][1], true, folderName);
        namedWindow( "INI", CV_WINDOW_AUTOSIZE );moveWindow("INI", 0, 0);
        Mat finImg = LoadImage(imageTriplets[i][2], true, folderName);
        namedWindow( "FIN", CV_WINDOW_AUTOSIZE );moveWindow("FIN", 800, 0);
        float contDist = getDistanceBetweenContours(inContours[i],finContours[i]);
        float contDistThres = 10;//TODO remove
        //draw everything on first image
        cout << " contDist is " << contDist << endl;
        cout << " toolDist is " <<  norm(finAction[i]-inAction[i]);

        if (contDist > contDistThres) {
            arrowedLine(iniImg, avgInContour, avgFinContour, Scalar(255,255,0));
        } else {
            arrowedLine(iniImg, avgInContour, avgFinContour, Scalar(0,0,0));
        }
        arrowedLine(iniImg, inAction[i], finAction[i], Scalar(0,255,255));
        for (uint j = 0; j < inContours[i].size(); j++) {
            stringstream strs;
            strs << j;
            string temp_str = strs.str();
            char* char_type = (char*) temp_str.c_str();
            circle(iniImg, inContours[i][j], 2, Scalar(255,0,0), -1);
            putText(iniImg, char_type, inContours[i][j], FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(255,0,0));
            circle(iniImg, finContours[i][j], 2, Scalar(0,0,255), -1);
            putText(iniImg, char_type, finContours[i][j], FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0,0,255));
        }
        writePathOnImage(iniImg, imageTriplets[i][0], imageTriplets[i][1]);
        imshow( "INI", iniImg);
        //draw everything on last image
        if (contDist > contDistThres) {
            arrowedLine(finImg, avgInContour, avgFinContour, Scalar(255,255,0));
        } else {
            arrowedLine(finImg, avgInContour, avgFinContour, Scalar(0,0,0));
        }
        arrowedLine(finImg, inAction[i], finAction[i], Scalar(0,255,255));
        for (uint j = 0; j < inContours[i].size(); j++) {
            circle(finImg, inContours[i][j], 2, Scalar(255,0,0), -1);
            circle(finImg, finContours[i][j], 2, Scalar(0,0,255), -1);
        }
        writePathOnImage(finImg, imageTriplets[i][0], imageTriplets[i][2]);
        imshow( "FIN", finImg);

        //if (i > 1116930) waitKey(0);
        //else waitKey(0);


        int key = waitKey();
        if (key == 1048696){//key "x"
            printf("\n\nsave this image\n\n");
            for (uint j = 0; j < inContours[j].size(); j++) {
                contoursFileManual << inContours[i][j].x << "\t" << inContours[i][j].y << "\t";
            }
            for (uint j = 0; j < finContours[j].size(); j++) {
                contoursFileManual << finContours[i][j].x << "\t" << finContours[i][j].y << "\t";
            }
            contoursFileManual  << endl;
            actionFileManual << inAction[i].x << "\t" << inAction[i].y << "\t" << finAction[i].x << "\t" << finAction[i].y;
            actionFileManual << endl;
        }
    }
    contoursFileManual.close();
    actionFileManual.close();
}
int main() {
    cout << "hello world" << endl;
    enum Step {trackTool, generateAndVisualizeDataForLearning};
    Step step = generateAndVisualizeDataForLearning;
    enum Action {push, tap, mix};
    Action action = push;
    const char * subsetsfileName;
    if (action == push) subsetsfileName = "../images/push.txt";
    else if (action == tap) subsetsfileName = "../images/tap.txt";
    else subsetsfileName = "../images/mix.txt";
    const bool showImages = false;
    switch (step) {
    case (trackTool): {
        vector <Vec3i> annotatedSequences = loadSubSets(subsetsfileName);
        getAndSaveCentroidsAndMotPrimSequences(annotatedSequences, showImages);
        break;
    }
    case (generateAndVisualizeDataForLearning): {
/*
        vector<Vec3i> motPrimSequenceCtr = loadMotPrimitives("motPrimPush.txt");
        vector <centroidInfo> allCentroidInfo = loadAllCentroidInfo("centroidFileProcessed.txt");
        vector <vector <centroidInfo> > allCentroidInfoBySequence;
        allCentroidInfoBySequence.resize(motPrimSequenceCtr.size());
        ofstream learningFile;
        learningFile.open("toolAndContour.txt");
        for (uint j = 0; j < motPrimSequenceCtr.size(); j++) {
            allCentroidInfoBySequence[j] = getCentroidInfoForSequence(motPrimSequenceCtr[j], allCentroidInfo);
            processImageSequencePush(motPrimSequenceCtr[j], allCentroidInfoBySequence[j], learningFile, showImages);//push
        }
        learningFile.close();
        generateDataForLearningPush("toolAndContour.txt", "U_push_images.txt", "U_push_contours.txt", "Y_push.txt");
*/
        if (true) {
            visualizeDataForLearningPush("U_push_images.txt", "U_push_contours.txt", "Y_push.txt");
        }
        break;
    }
    default: {
        cout << "specify a step " << endl;
        break;
    }
    }
    cout << "*********** MAIN END!! **********" << endl;
    return 0;
}
/*
Mat SegmentHSVwithMask(const Mat & srcImg, const Mat & mask, const Scalar & minHSV, const Scalar & maxHSV){
    // convert image to hsv
    Mat hsv_image;
    if(hsv_image.data == NULL){
        hsv_image.create(srcImg.size(), srcImg.type());
    }
    cvtColor(srcImg, hsv_image, CV_RGB2HSV);
    //segment according to levels
    Mat segImg;
    inRange(hsv_image, minHSV, maxHSV, segImg);
    //apply a mask on the tool
    if (mask.data != NULL) {
        Mat dstImg;
        bitwise_and(segImg, mask, dstImg);
        return(dstImg);
    } else {
        return(segImg);
    }
}
void annotateImages(const int & startCnt, const int & endCnt, const string & fName) {
    Mat imageRGB;
    int cntImg = startCnt;
    namedWindow( "CUR", CV_WINDOW_AUTOSIZE );moveWindow("CUR", 0, 0);
    int key = waitKey();
    while (key != 1048696) {      //key "x"
        //while not key=esc get keyboard
        if (key == 1113939) cntImg++;//key right arrow
        if (key == 1113937) cntImg--;//key left arrow
        if ((key == 1113937)||(key == 1113939)) {
            if ((cntImg >= startCnt) && (cntImg <= endCnt)) {
                imageRGB = LoadImage(cntImg, true, fName);
                ostringstream ossCnt;
                ossCnt << setw(4) << setfill('0') << cntImg;
                string imageNum = ossCnt.str();
                putText(imageRGB, imageNum, Point(50,55), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255));
                imshow( "CUR", imageRGB);
                waitKey(5);
            } else {
                cout << "wrong cnt - should be between " << startCnt << " and " << endCnt << " instead is " << cntImg << endl;
            }
        }
        key = waitKey();
    }
}
Mat DrawSandImage(const Mat & binSandImage, const Mat & image){
    Mat sandImage;
    Mat bgrSandImage;
    cvtColor(binSandImage, bgrSandImage, CV_GRAY2BGR);
    bitwise_and(image, bgrSandImage, sandImage);
    return(sandImage);
}
Mat GetBinaryROIimage(const Mat & image, const Rect & largest_bounding_rect){
    Mat roiImage;
    Mat roi_mask = Mat::zeros(image.rows, image.cols, image.type());
    roi_mask(largest_bounding_rect) = Scalar(255,255,255);
    bitwise_and(image, roi_mask, roiImage);
    return (roiImage);
}
void processImageSequenceIncide(const int & startCnt, const int & endCnt, const string & fName) {
    Mat startImg = LoadImage(startCnt, true, fName);
    namedWindow( "START", CV_WINDOW_AUTOSIZE );moveWindow("START", 0, 0);imshow( "START", startImg);
    Mat endImg = LoadImage(endCnt, true, fName);
    namedWindow( "END", CV_WINDOW_AUTOSIZE );moveWindow("END", 700, 0);imshow( "END", endImg);

    Mat grayErrImg;
    subtract(startImg, endImg, grayErrImg);
    cvtColor( grayErrImg, grayErrImg, CV_BGR2GRAY );
    threshold(grayErrImg, grayErrImg, 20, 255, THRESH_BINARY);
    grayErrImg = MorphologyOperations(grayErrImg, 3, 3, 0);
    vector< vector<Point> > bigCont = GetBigContours(grayErrImg, 100);
    if (bigCont.size()==0) cout << "in processImageSequenceIncide no bigCont"<<endl;
    for(unsigned int i = 0; i< bigCont.size(); i++ ) {
        for(unsigned int j = 0; j< bigCont.at(i).size(); j++ ) {
            circle(endImg, bigCont.at(i).at(j), 1, Scalar(0,0,255));
        }
    }
    namedWindow( "END", CV_WINDOW_AUTOSIZE );moveWindow("END", 1000, 0);imshow( "END", endImg);
    waitKey(0);
}
void processImageSequenceTap(const int & startCnt, const int & endCnt, const string & fName) {
    static const int binThrTap = 5;
    Mat startImg = LoadImage(startCnt, true, fName);
    namedWindow( "START", CV_WINDOW_AUTOSIZE );moveWindow("START", 0, 0);imshow( "START", startImg);
    Mat endImg = LoadImage(endCnt, true, fName);
    namedWindow( "END", CV_WINDOW_AUTOSIZE );moveWindow("END", 700, 0);imshow( "END", endImg);

    Mat grayErrImg;
    subtract(endImg, startImg, grayErrImg);

    cvtColor( grayErrImg, grayErrImg, CV_BGR2GRAY );
    threshold(grayErrImg, grayErrImg, binThrTap, 255, THRESH_BINARY);
    namedWindow( "BIN", CV_WINDOW_AUTOSIZE );moveWindow("BIN", 0, 900);imshow( "BIN", grayErrImg);
    grayErrImg = MorphologyOperations(grayErrImg, 3, 1, 0);
    namedWindow( "MORPH", CV_WINDOW_AUTOSIZE );moveWindow("MORPH", 700, 900);imshow( "MORPH", grayErrImg);
    vector< vector<Point> > bigCont = GetBigContours(grayErrImg, 500);
    if (bigCont.size()==0) cout << "in processImageSequenceTap no bigCont"<<endl;
    Rect PushStartToEndDiffROI = MergeBoundingBoxes(bigCont, 0.6);
    cout << "br " << PushStartToEndDiffROI.br().x << " " << PushStartToEndDiffROI.br().y << " tl " << PushStartToEndDiffROI.tl().x << " " << PushStartToEndDiffROI.tl().y << " " <<endl;
    vector< Rect> allRoi;

    static const uint rowStride = 40;
    static const uint colStride = 40;

    static const uint allRoiRows = (PushStartToEndDiffROI.br().y - PushStartToEndDiffROI.tl().y)/rowStride;
    static const uint allRoiCols = (PushStartToEndDiffROI.br().x - PushStartToEndDiffROI.tl().x)/colStride;

    cout << "strides R " << rowStride << " C " << colStride << endl;
    cout << "all R " << allRoiRows << " C " << allRoiCols << endl;
    for (uint i =0; i<allRoiCols; i++) {
        for (uint j =0; j<allRoiRows; j++) {
            Rect thisRoi(PushStartToEndDiffROI.tl().x+i*colStride, PushStartToEndDiffROI.tl().y+j*rowStride, colStride, rowStride);
            allRoi.push_back(thisRoi);
        }
    }
    vector <double> oldStddev;
    for (uint i =0; i<allRoiCols*allRoiRows; i++) {
        rectangle(endImg, allRoi.at(i), Scalar(0,255,0));
        oldStddev.push_back(-1.);
    }
    //draw roi in desired image
    namedWindow( "END", CV_WINDOW_AUTOSIZE );moveWindow("END", 1000, 0);imshow( "END", endImg);
    //LOOP
    Mat imageRGB;
    Mat imageGray;
    //Mat imageDepth;
    vector <vector <double> > stddev;
    stddev.resize(allRoiCols*allRoiRows);
    ofstream tapFile;
    tapFile.open ("tapFile.txt");
    static const double blackThreshold = 120;
    for (int cntImg = startCnt; cntImg <= endCnt; cntImg++) {
        imageRGB = LoadImage(cntImg, true, fName);
        //imageDepth = LoadImage(cntImg, false, fName);
        cvtColor(imageRGB, imageGray, CV_BGR2GRAY);
        for (uint i =0; i<allRoiCols*allRoiRows; i++) {
            Mat thisRoi = imageGray(allRoi.at(i));
            Scalar m, s;
            meanStdDev(thisRoi, m, s);
            if ((double)m.val[0] < blackThreshold) {
                stddev[i].push_back(oldStddev[i]);
                rectangle(imageRGB, allRoi.at(i), Scalar(0,0,255));
            } else {
                if ((oldStddev[i] == -1) && (cntImg > startCnt)) {
                    for (int j = 0; j < cntImg - startCnt + 1; j++) {
                        stddev[i][j] = ((double)s.val[0]);
                    }
                }
                stddev[i].push_back((double)s.val[0]);
                rectangle(imageRGB, allRoi.at(i), Scalar(0,255,0));
            }
            oldStddev[i] = stddev[i].back();
        }
        namedWindow( "CUR", CV_WINDOW_AUTOSIZE );moveWindow("CUR", 0, 0);imshow( "CUR", imageRGB);
        //namedWindow( "DEPTHIMAGE", CV_WINDOW_AUTOSIZE );moveWindow("DEPTHIMAGE", 1000, 0);imshow( "DEPTHIMAGE", imageDepth);
        waitKey(50);
    }
    for (int j = 0; j <= endCnt - startCnt; j++) {
        for (uint i =0; i<allRoiCols*allRoiRows; i++) {
            tapFile << stddev[i][j] << "\t";
        }
        tapFile << endl;
    }
    cout << "normalized difference is ";
    int pos=0, neg=0, zero=0;
    for (uint i =0; i<allRoiCols*allRoiRows; i++) {
        double normStDev = (double)(stddev[i][endCnt - startCnt-1] - stddev[i][0])/ (double)(stddev[i][0]);
        cout << normStDev << "\t";
        if (normStDev == 0) zero++; else if (normStDev > 0) pos++; else if (normStDev < 0) neg++;
    }
    cout << endl << "pos " << pos <<" neg " << neg << " zero " << zero ;
    tapFile << endl;
    tapFile.close();
}
*/
