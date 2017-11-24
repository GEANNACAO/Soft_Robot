// StereoCalibration.cpp : �������̨Ӧ�ó������ڵ㡣

#include "stdafx.h"

//�ڽ���˫Ŀ����ͷ�ı궨֮ǰ��������ȷֱ����������ͷ���е�Ŀ�Ӿ��ı궨 
//�ֱ�ȷ����������ͷ���ڲξ���Ȼ���ٿ�ʼ����˫Ŀ����ͷ�ı궨
//�ڴ����������ȶ���������ͷ���е����궨(����һƪ��Ŀ�궨����)��Ȼ���ڽ�������궨

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

const int imageWidth =2592;								//����ͷ�ķֱ���
const int imageHeight = 2048;
const int boardWidth = 11;								//����Ľǵ���Ŀ
const int boardHeight = 8;								//����Ľǵ�����
const int boardCorner = boardWidth * boardHeight;		//�ܵĽǵ�����
const int frameNumber = 18;								//����궨ʱ��Ҫ���õ�ͼ��֡��
const int squareSize = 30;								//�궨��ڰ׸��ӵĴ�С ��λmm
const Size boardSize = Size(boardWidth, boardHeight);	//
Size imageSize = Size(imageWidth, imageHeight);

Mat R, T, E, F;											//R ��תʸ�� Tƽ��ʸ�� E�������� F��������
vector<Mat> rvecs;									    //��ת����
vector<Mat> tvecs;										//ƽ������
vector<vector<Point2f>> imagePointL;				    //��������������Ƭ�ǵ�����꼯��
vector<vector<Point2f>> imagePointR;					//�ұ������������Ƭ�ǵ�����꼯��
vector<vector<Point3f>> objRealPoint;					//����ͼ��Ľǵ��ʵ���������꼯��

vector<Point2f> cornerL;								//��������ĳһ��Ƭ�ǵ����꼯��
vector<Point2f> cornerR;								//�ұ������ĳһ��Ƭ�ǵ����꼯��

Mat rgbImageL, grayImageL;                              //��ͼ���ɫͼ���Ҷ�ͼ
Mat rgbImageR, grayImageR;                              //��ͼ���ɫͼ���Ҷ�ͼ

Mat Rl, Rr, Pl, Pr, Q;									//У����ת����R��ͶӰ����P ��ͶӰ����Q (�����о���ĺ�����ͣ�	
Mat mapLx, mapLy, mapRx, mapRy;							//ӳ���
Rect validROIL, validROIR;								//ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������

/*
���ȱ궨�õ���������ڲξ���
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 2589.19600167291, 0, 1294.19718230902,
	                                       0, 2589.57099242041, 1010.12331158231,
	                                             0, 0 ,		1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.130324085910710, 0.129145703551819, -0.000363277602896836	,0.00100777334479341, -0.0159849186063040);       //���������ϵ��
/*
���ȱ궨�õ���������ڲξ���
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixR = (Mat_<double>(3, 3) << 2585.22844438238, 0, 1333.96627119011,
	                                      0	,2587.13438230198,1058.31938909946,
			                             0,    0,     1);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.140414252361196, 0.222330025868883, -0.00100785606954588	,0.00121895260422862, -0.252334054806197);


/*����궨����ģ���ʵ����������,��λmm*/
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
	/*��������*/
	/*�������*/
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


//��������֮�󱣴�����,��Ҫͷ�ļ� #include <fstream>
static void saveXYZ(string filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	ofstream fp(filename);
	if (!fp.is_open())
	{
		std::cout << "�򿪵����ļ�ʧ��" << endl;
		fp.close();
		return;
	}
	//����д��
	for (int y = 0; y < mat.rows;y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);//��ͨ��������
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
	cout << "��Q�˳� ..." << endl;
	while (goodFrameCount < frameNumber)
	{
		char filename[100];

		/*��ȡ��ߵ�ͼ��*/
		sprintf_s(filename, "image20170605\\left%02d.bmp", goodFrameCount + 1);    //��ȡimage�ļ����µ�ͼƬ
		rgbImageL = imread(filename, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);

		/*��ȡ�ұߵ�ͼ��*/

		sprintf_s(filename, "image20170605\\right%02d.bmp", goodFrameCount + 1);
		rgbImageR = imread(filename, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

		bool isFindL, isFindR;

		isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		if (isFindL == true && isFindR == true)	 //�������ͼ���ҵ������еĽǵ� ��˵��������ͼ���ǿ��е�
		{
			/*
			Size(5,5) �������ڵ�һ���С
			Size(-1,-1) ������һ��ߴ�
			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 0.01)������ֹ����
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
			����Ӧ���ж�������ͼ���ǲ��Ǻõģ��������ƥ��Ļ��ſ��������궨
			������������̵��У��õ�ͼ����ϵͳ�Դ���ͼ�񣬶��ǿ���ƥ��ɹ��ġ�
			���������û���ж�
			*/
			//string filename = "res\\image\\calibration";
			//filename += goodFrameCount + ".jpg";
			//cvSaveImage(filename.c_str(), &IplImage(rgbImage));		//�Ѻϸ��ͼƬ��������
			goodFrameCount++;
			cout << "The image is good!" << endl;
		}
		else
		{
			cout << "The image is bad ,please try again" << endl;
		}

	}
		/*
		����ʵ�ʵ�У�������ά����
		����ʵ�ʱ궨���ӵĴ�С������
		*/
		calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
		cout << "cal real successful" << endl;

		/*
		�궨����ͷ
		��������������ֱ𶼾����˵�Ŀ�궨
		�����ڴ˴�ѡ��flag = CALIB_USE_INTRINSIC_GUESS
		*/
		double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
			cameraMatrixL, distCoeffL,
			cameraMatrixR, distCoeffR,
			Size(imageWidth, imageHeight), R, T, E, F,	
			TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-9), CALIB_FIX_INTRINSIC);

		cout << "Stereo Calibration done with RMS error = " << rms << endl;

		////����У��
		/*
		����У����ʱ����Ҫ����ͼ���沢���ж�׼ ��ʹ������ƥ����ӵĿɿ�
		ʹ������ͼ����ķ������ǰ���������ͷ��ͼ��ͶӰ��һ�������������ϣ�����ÿ��ͼ��ӱ�ͼ��ƽ��ͶӰ������ͼ��ƽ�涼��Ҫһ����ת����R
		stereoRectify���� �����������ľ��Ǵ�ͼ��ƽ��ͶӰ����������ƽ�����ת����Rl,Rr�� Rl,Rr��Ϊ�������ƽ���ж�׼��У����ת����
		���������Rl��ת�����������Rr��ת֮������ͼ����Ѿ����沢���ж�׼�ˡ�
		����Pl,PrΪ���������ͶӰ�����������ǽ�3D�������ת����ͼ���2D�������:P*[X Y Z 1]' =[x y w]
		Q����Ϊ��ͶӰ���󣬼�����Q���԰�2άƽ��(ͼ��ƽ��)�ϵĵ�ͶӰ��3ά�ռ�ĵ�:Q*[x y d 1] = [X Y Z W]������dΪ��������ͼ����Ӳ�
		*/
		stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);




		////////////////////////////////////////////////////////////////////////////////////
		////===================���ͼ���еĽǵ�����������Ϣ=================================		
		cout << "��ͼ�������й���" << imagePointL.size() << "��ͼ��" << endl;
		for (int i = 0; i < imagePointL.size(); i++)
		{			
			cout << "��ͼ�������е�" <<i+1<<"��ͼ��Ľǵ�����Ϊ��"<< imagePointL.at(i).size() << "��" << endl;   //��ʾ��ͼ�������е�i��ͼ��Ľǵ�����
			cout << "��"<<i+1<<"��ͼ������нǵ�����Ϊ��\n" << imagePointL.at(i) <<"\n"<< endl; //ͼ�������нǵ��������� 
		}
		
		cout << "��ͼ�������й���" << imagePointR.size() << "��ͼ��" << endl;
		for (int i = 0; i < imagePointR.size(); i++)
		{
			cout << "��ͼ�������е�" << i + 1 << "��ͼ��Ľǵ�����Ϊ��" << imagePointR.at(i).size() << "��" << endl;   //��ʾ��ͼ�������е�i��ͼ��Ľǵ�����
			cout << "��" << i + 1 << "��ͼ������нǵ�����Ϊ��\n" << imagePointR.at(i) << "\n" << endl; //ͼ�������нǵ��������� 
		}
		///////////////////////////////////////////////////////////////////////////////////
		
		/*
		����stereoRectify ���������R �� P ������ͼ���ӳ��� mapx,mapy
		mapx,mapy������ӳ�����������Ը�remap()�������ã���У��ͼ��ʹ������ͼ���沢���ж�׼
		ininUndistortRectifyMap()�Ĳ���newCameraMatrix����У����������������openCV���棬У����ļ��������Mrect�Ǹ�ͶӰ����Pһ�𷵻صġ�
		�������������ﴫ��ͶӰ����P���˺������Դ�ͶӰ����P�ж���У��������������
		*/
		//��ȡ������Ľ���ӳ��
		initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
		initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);


		Mat rectifyImageL, rectifyImageR;
		cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
		cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);
		resize(rectifyImageL, rectifyImageL, Size(), 0.3, 0.3);
		imshow("Rectify Before", rectifyImageL);

		/*
		����remap֮�����������ͼ���Ѿ����沢���ж�׼��
		*/
		//����ԭʼͼ��
		remap(rectifyImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
		remap(rectifyImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

		

		/*���沢�������*/
		outputCameraParam();
	

		/////////////////////////////////////////////////////////////////////////////////////////////////
		///����ƥ���㷨��BM�㷨��
		////================�����Ӳ�ͼ����ͶӰ3D����===================================
		cv::Mat imgDisparity32F = Mat(rectifyImageL.rows, rectifyImageL.cols, CV_32F);
		StereoBM sbm(StereoBM::BASIC_PRESET, 80, 5);          //ƥ������㷨����
		// Ԥ�����˲�����
		sbm.state->preFilterSize = 15;   //Ԥ�����˲����Ĵ��ڴ�С
		sbm.state->preFilterCap = 20;    //Ԥ�����˲����Ľض�ֵ
		sbm.state->SADWindowSize = 11;   //SAD���ڴ�С,Sum of absolute differrences
		sbm.state->minDisparity = 0;     //��С�ӲĬ��ֵΪ 0
		sbm.state->numberOfDisparities = 80;   //�Ӳ��
		sbm.state->textureThreshold = 0;        //������������ж���ֵ
		sbm.state->uniquenessRatio = 8;         //�Ӳ�Ψһ�԰ٷֱ�
		sbm.state->speckleWindowSize = 0;       //����Ӳ���ͨ����仯�ȵĴ��ڴ�С��ֵΪ0ʱȡ��speckle���
		sbm.state->speckleRange = 0;            //�Ӳ�仯��ֵ����������ʱ��仯������ֵʱ���ô����ڵ��Ӳ����㣬int��
		 
		Mat dispImageL, dispImageR;
		cvtColor(rectifyImageL, dispImageL, CV_BGR2GRAY);  //ת��Ϊ8λ��ͨ���Ҷ�ͼ��
		cvtColor(rectifyImageR, dispImageR, CV_BGR2GRAY);

		// �����Ӳ�
		sbm(dispImageL, dispImageR, imgDisparity32F, CV_32F);

		//���Ӳ�ͼ������������
		cv::Mat_<cv::Vec3f>  XYZ(imgDisparity32F.size(), CV_32FC1);    //������ƣ�X,Y,Z��
		reprojectImageTo3D(imgDisparity32F, XYZ, Q, true);
		//Q����Ϊ��ͶӰ���󣬼�����Q���԰�2άƽ��(ͼ��ƽ��)�ϵĵ�ͶӰ��3ά�ռ�ĵ�:Q*[x y d 1] = [X Y Z W]������dΪ��������ͼ����Ӳ�
		//�ռ��ʵ����ά���꣨x,y,z��=��X/W,Y/W,Z/W��
		/*cv::destroyAllWindows();
		cout << endl << "����ĵ�������..." << endl;
		saveXYZ(point_cloud_filename, XYZ);
*/
	
		imshow("�Ӳ�ͼ",imgDisparity32F);
		////////////////////////////////////////////////////////////////////////////////////////////////////


		/*��У�������ʾ����
		����������ͼ����ʾ��ͬһ��������
		����ֻ��ʾ�����һ��ͼ���У���������û�а����е�ͼ����ʾ����
		*/
		Mat canvas;             //��ʾ����
		double sf;              //���ű�������
		int w, h;               //����һ���Ⱥͻ����߶�
		sf = 1296. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width * sf);             //cvRound()����һ��double�͵���������������
		h = cvRound(imageSize.height * sf);
		canvas.create(h, w * 2, CV_8UC3);              //�����������߶ȡ���ȡ�����

		/*��ͼ�񻭵�������*/
		Mat canvasPart = canvas(Rect(w * 0, 0, w, h));								//�õ�������һ����
		resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);		//��ͼ�����ŵ���canvasPartһ����С
		Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf), cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));   //��ñ���ȡ������
		rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);						//����һ������

		cout << "Painted ImageL" << endl;

		/*��ͼ�񻭵�������*/
		canvasPart = canvas(Rect(w, 0, w, h));										//��û�������һ����
		resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
		Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf), cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
		rectangle(canvasPart, vroiR, Scalar(0, 255, 255), 3, 8);

		cout << "Painted ImageR" << endl;

		/*���϶�Ӧ������*/
		for (int i = 0; i < canvas.rows; i += 30)
			line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

		imshow("rectified", canvas);

		cout << "wait key" << endl;
		waitKey(0);
		system("pause");
		return 0;
}
