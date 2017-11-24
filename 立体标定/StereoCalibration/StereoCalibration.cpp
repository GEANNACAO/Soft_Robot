// StereoCalibration.cpp : 定义控制台应用程序的入口点。

#include "stdafx.h"

//在进行双目摄像头的标定之前，最好事先分别对两个摄像头进行单目视觉的标定 
//分别确定两个摄像头的内参矩阵，然后再开始进行双目摄像头的标定
//在此例程中是先对两个摄像头进行单独标定(见上一篇单目标定文章)，然后在进行立体标定

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cv.h"
#include <cv.hpp>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace std;
using namespace cv;

const int imageWidth =2592;								//摄像头的分辨率
const int imageHeight = 2048;
const int boardWidth = 11;								//横向的角点数目
const int boardHeight = 8;								//纵向的角点数据
const int boardCorner = boardWidth * boardHeight;		//总的角点数据
const int frameNumber = 18;								//相机标定时需要采用的图像帧数
const int squareSize = 30;								//标定板黑白格子的大小 单位mm
const Size boardSize = Size(boardWidth, boardHeight);	//
Size imageSize = Size(imageWidth, imageHeight);

Mat R, T, E, F;											//R 旋转矢量 T平移矢量 E本征矩阵 F基础矩阵
vector<Mat> rvecs;									    //旋转向量
vector<Mat> tvecs;										//平移向量
vector<vector<Point2f>> imagePointL;				    //左边摄像机所有照片角点的坐标集合
vector<vector<Point2f>> imagePointR;					//右边摄像机所有照片角点的坐标集合
vector<vector<Point3f>> objRealPoint;					//各副图像的角点的实际物理坐标集合

vector<Point2f> cornerL;								//左边摄像机某一照片角点坐标集合
vector<Point2f> cornerR;								//右边摄像机某一照片角点坐标集合

Mat rgbImageL, grayImageL;                              //左图像彩色图、灰度图
Mat rgbImageR, grayImageR;                              //右图像彩色图、灰度图

Mat Rl, Rr, Pl, Pr, Q;									//校正旋转矩阵R，投影矩阵P 重投影矩阵Q (下面有具体的含义解释）	
Mat mapLx, mapLy, mapRx, mapRy;							//映射表
Rect validROIL, validROIR;								//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域

/*
事先标定好的左相机的内参矩阵
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 2589.19600167291, 0, 1294.19718230902,
	                                       0, 2589.57099242041, 1010.12331158231,
	                                             0, 0 ,		1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.130324085910710, 0.129145703551819, -0.000363277602896836	,0.00100777334479341, -0.0159849186063040);       //左相机畸变系数
/*
事先标定好的右相机的内参矩阵
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixR = (Mat_<double>(3, 3) << 2585.22844438238, 0, 1333.96627119011,
	                                      0	,2587.13438230198,1058.31938909946,
			                             0,    0,     1);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.140414252361196, 0.222330025868883, -0.00100785606954588	,0.00121895260422862, -0.252334054806197);


/*计算标定板上模块的实际物理坐标,单位mm*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
{
	//	Mat imgpoint(boardheight, boardwidth, CV_32FC3,Scalar(0,0,0));
	vector<Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
	{
		for (int colIndex = 0; colIndex < boardwidth; colIndex++)
		{
			//	imgpoint.at<Vec3f>(rowIndex, colIndex) = Vec3f(rowIndex * squaresize, colIndex*squaresize, 0);
			imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
}

void outputCameraParam(void)
{
	/*保存数据*/
	/*输出数据*/
	FileStorage fs("intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
		fs.release();
		cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
	}
	else
	{
		cout << "Error: can not save the intrinsics!!!!!" << endl;
	}

	fs.open("extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q;
		cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";
}


//生产点云之后保存坐标,需要头文件 #include <fstream>
static void saveXYZ(string filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	ofstream fp(filename);
	if (!fp.is_open())
	{
		std::cout << "打开点云文件失败" << endl;
		fp.close();
		return;
	}
	//遍历写入
	for (int y = 0; y < mat.rows;y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);//三通道浮点型
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
				continue;
			fp << point[0] << "" << point[1] <<"" << point[2] << endl;
		}
	}
	fp.close();
}


int _tmain(int argc, _TCHAR* argv[])
{
	Mat img;
	int goodFrameCount = 0;
	/*namedWindow("ImageL");
	namedWindow("ImageR");*/
	cout << "按Q退出 ..." << endl;
	while (goodFrameCount < frameNumber)
	{
		char filename[100];

		/*读取左边的图像*/
		sprintf_s(filename, "image20170605\\left%02d.bmp", goodFrameCount + 1);    //读取image文件夹下的图片
		rgbImageL = imread(filename, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);

		/*读取右边的图像*/

		sprintf_s(filename, "image20170605\\right%02d.bmp", goodFrameCount + 1);
		rgbImageR = imread(filename, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

		bool isFindL, isFindR;

		isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		if (isFindL == true && isFindR == true)	 //如果两幅图像都找到了所有的角点 则说明这两幅图像是可行的
		{
			/*
			Size(5,5) 搜索窗口的一半大小
			Size(-1,-1) 死区的一半尺寸
			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 0.01)迭代终止条件
			*/
			cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10000, 0.0001));
			drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);			
			
			//resize(rgbImageL, rgbImageL, Size(), 0.3, 0.3);
			imshow("chessboardL", rgbImageL);
			imagePointL.push_back(cornerL);


			cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER,10000, 0.0001));
			drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
			
			/*resize(rgbImageR, rgbImageR, Size(), 0.3, 0.3);*/
			imshow("chessboardR", rgbImageR);
			imagePointR.push_back(cornerR);

			/*
			本来应该判断这两幅图像是不是好的，如果可以匹配的话才可以用来标定
			但是在这个例程当中，用的图像是系统自带的图像，都是可以匹配成功的。
			所以这里就没有判断
			*/
			//string filename = "res\\image\\calibration";
			//filename += goodFrameCount + ".jpg";
			//cvSaveImage(filename.c_str(), &IplImage(rgbImage));		//把合格的图片保存起来
			goodFrameCount++;
			cout << "The image is good!" << endl;
		}
		else
		{
			cout << "The image is bad ,please try again" << endl;
		}

	}
		/*
		计算实际的校正点的三维坐标
		根据实际标定格子的大小来设置
		*/
		calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
		cout << "cal real successful" << endl;

		/*
		标定摄像头
		由于左右摄像机分别都经过了单目标定
		所以在此处选择flag = CALIB_USE_INTRINSIC_GUESS
		*/
		double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
			cameraMatrixL, distCoeffL,
			cameraMatrixR, distCoeffR,
			Size(imageWidth, imageHeight), R, T, E, F,	
			TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-9), CALIB_FIX_INTRINSIC);

		cout << "Stereo Calibration done with RMS error = " << rms << endl;

		////立体校正
		/*
		立体校正的时候需要两幅图像共面并且行对准 以使得立体匹配更加的可靠
		使得两幅图像共面的方法就是把两个摄像头的图像投影到一个公共成像面上，这样每幅图像从本图像平面投影到公共图像平面都需要一个旋转矩阵R
		stereoRectify（） 这个函数计算的就是从图像平面投影到公共成像平面的旋转矩阵Rl,Rr。 Rl,Rr即为左右相机平面行对准的校正旋转矩阵。
		左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。
		其中Pl,Pr为两个相机的投影矩阵，其作用是将3D点的坐标转换到图像的2D点的坐标:P*[X Y Z 1]' =[x y w]
		Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。其中d为左右两幅图像的视差
		*/
		stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);




		////////////////////////////////////////////////////////////////////////////////////
		////===================输出图像中的角点像素坐标信息=================================		
		cout << "左图像序列中共有" << imagePointL.size() << "幅图像" << endl;
		for (int i = 0; i < imagePointL.size(); i++)
		{			
			cout << "左图像序列中第" <<i+1<<"幅图像的角点总数为："<< imagePointL.at(i).size() << "个" << endl;   //显示做图像序列中第i副图像的角点总数
			cout << "第"<<i+1<<"幅图像的所有角点坐标为：\n" << imagePointL.at(i) <<"\n"<< endl; //图像中所有角点像素坐标 
		}
		
		cout << "右图像序列中共有" << imagePointR.size() << "幅图像" << endl;
		for (int i = 0; i < imagePointR.size(); i++)
		{
			cout << "右图像序列中第" << i + 1 << "幅图像的角点总数为：" << imagePointR.at(i).size() << "个" << endl;   //显示做图像序列中第i副图像的角点总数
			cout << "第" << i + 1 << "幅图像的所有角点坐标为：\n" << imagePointR.at(i) << "\n" << endl; //图像中所有角点像素坐标 
		}
		///////////////////////////////////////////////////////////////////////////////////
		
		/*
		根据stereoRectify 计算出来的R 和 P 来计算图像的映射表 mapx,mapy
		mapx,mapy这两个映射表接下来可以给remap()函数调用，来校正图像，使得两幅图像共面并且行对准
		ininUndistortRectifyMap()的参数newCameraMatrix就是校正后的摄像机矩阵。在openCV里面，校正后的计算机矩阵Mrect是跟投影矩阵P一起返回的。
		所以我们在这里传入投影矩阵P，此函数可以从投影矩阵P中读出校正后的摄像机矩阵
		*/
		//获取两相机的矫正映射
		initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
		initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);


		Mat rectifyImageL, rectifyImageR;
		cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
		cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);
		resize(rectifyImageL, rectifyImageL, Size(), 0.3, 0.3);
		imshow("Rectify Before", rectifyImageL);

		/*
		经过remap之后，左右相机的图像已经共面并且行对准了
		*/
		//矫正原始图像
		remap(rectifyImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
		remap(rectifyImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

		

		/*保存并输出数据*/
		outputCameraParam();
	

		/////////////////////////////////////////////////////////////////////////////////////////////////
		///立体匹配算法（BM算法）
		////================计算视差图，重投影3D坐标===================================
		cv::Mat imgDisparity32F = Mat(rectifyImageL.rows, rectifyImageL.cols, CV_32F);
		StereoBM sbm(StereoBM::BASIC_PRESET, 80, 5);          //匹配代价算法类型
		// 预处理滤波参数
		sbm.state->preFilterSize = 15;   //预处理滤波器的窗口大小
		sbm.state->preFilterCap = 20;    //预处理滤波器的截断值
		sbm.state->SADWindowSize = 11;   //SAD窗口大小,Sum of absolute differrences
		sbm.state->minDisparity = 0;     //最小视差，默认值为 0
		sbm.state->numberOfDisparities = 80;   //视差窗口
		sbm.state->textureThreshold = 0;        //低纹理区域的判断阈值
		sbm.state->uniquenessRatio = 8;         //视差唯一性百分比
		sbm.state->speckleWindowSize = 0;       //检查视差连通区域变化度的窗口大小，值为0时取消speckle检查
		sbm.state->speckleRange = 0;            //视差变化阈值，当窗口内时差变化大于阈值时，该窗口内的视差清零，int型
		 
		Mat dispImageL, dispImageR;
		cvtColor(rectifyImageL, dispImageL, CV_BGR2GRAY);  //转化为8位单通道灰度图像
		cvtColor(rectifyImageR, dispImageR, CV_BGR2GRAY);

		// 计算视差
		sbm(dispImageL, dispImageR, imgDisparity32F, CV_32F);

		//从视差图计算世界坐标
		cv::Mat_<cv::Vec3f>  XYZ(imgDisparity32F.size(), CV_32FC1);    //输出点云（X,Y,Z）
		reprojectImageTo3D(imgDisparity32F, XYZ, Q, true);
		//Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。其中d为左右两幅图像的视差
		//空间点实际三维坐标（x,y,z）=（X/W,Y/W,Z/W）
		/*cv::destroyAllWindows();
		cout << endl << "保存的点云坐标..." << endl;
		saveXYZ(point_cloud_filename, XYZ);
*/
	
		imshow("视差图",imgDisparity32F);
		////////////////////////////////////////////////////////////////////////////////////////////////////


		/*把校正结果显示出来
		把左右两幅图像显示到同一个画面上
		这里只显示了最后一副图像的校正结果。并没有把所有的图像都显示出来
		*/
		Mat canvas;             //显示画布
		double sf;              //缩放比例因子
		int w, h;               //画布一半宽度和画布高度
		sf = 1296. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width * sf);             //cvRound()：对一个double型的数进行四舍五入
		h = cvRound(imageSize.height * sf);
		canvas.create(h, w * 2, CV_8UC3);              //创建画布：高度、宽度、类型

		/*左图像画到画布上*/
		Mat canvasPart = canvas(Rect(w * 0, 0, w, h));								//得到画布的一部分
		resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);		//把图像缩放到跟canvasPart一样大小
		Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf), cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));   //获得被截取的区域
		rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);						//画上一个矩形

		cout << "Painted ImageL" << endl;

		/*右图像画到画布上*/
		canvasPart = canvas(Rect(w, 0, w, h));										//获得画布的另一部分
		resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
		Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf), cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
		rectangle(canvasPart, vroiR, Scalar(0, 255, 255), 3, 8);

		cout << "Painted ImageR" << endl;

		/*画上对应的线条*/
		for (int i = 0; i < canvas.rows; i += 30)
			line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

		imshow("rectified", canvas);

		cout << "wait key" << endl;
		waitKey(0);
		system("pause");
		return 0;
}
