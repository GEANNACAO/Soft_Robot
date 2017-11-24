#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
int main()
{
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//************************************************************************
	//	   从摄像机中读取多幅图像,从中提取出角点，然后对角点进行亚像素精确化
	//	   *************************************************************************/
	//int image_count = 10;                    /****    图像数量     ****/
	//Mat frame;
	//Size image_size;                         /****     图像的尺寸      ****/
	//Size board_size = Size(8, 5);            /****    定标板上每行、列的角点数 ****///不同标定板size也是不同的，根据自己的标定板更改括号内的值。
	//vector<Point2f> corners;                  /****    缓存每幅图像上检测到的角点       ****/
	//vector<vector<Point2f>>  corners_Seq;    /****  保存检测到的所有角点       ****/
	//ofstream fout("calibration_result.txt");  /**    保存定标结果的文件     **/
	//int mode = CAPTURING;
	//
	//VideoCapture cap(1);                                  //打开外部摄像机
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 1292);               // 相机的分辨率  
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 964);
	//if (!cap.isOpened()){
	//	std::cout << "打开摄像头失败，退出";
	//	exit(-1);
	//}
	//namedWindow("Calibration");
	//std::cout << "Press 'g' to start capturing images!" << endl;
	//
	//int count = 0, n = 0;
	//stringstream tempname;
	//string filename;
	//int key;
	//string msg;
	//int baseLine;
	//Size textSize;
	//
	//
	//while (n < image_count)
	//{
	//	frame.setTo(0);
	//	cap >> frame;
	//	if (mode == DETECTION)
	//	{
	//		key = 0xff & waitKey(30);
	//		if ((key & 255) == 27)
	//			break;
	//
	//		if (cap.isOpened() && key == 'g')
	//		{
	//			mode = CAPTURING;
	//		}
	//	}
	//
	//	if (mode == CAPTURING)
	//	{
	//		int kk = waitKey(30);
	//		key = 0xff & kk;
	//		/*if ((key & 255) == 32)
	//		{*/
	//			image_size = frame.size();
	//			/* 提取角点 */
	//			Mat imageGray;
	//			cvtColor(frame, imageGray , CV_RGB2GRAY);
	//			bool patternfound = findChessboardCorners(frame, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	//			if (patternfound)
	//			{
	//				n++;
	//				tempname << n;
	//				tempname >> filename;
	//				filename += ".jpg";
	//				/* 亚像素精确化 */
	//				cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	//				count += corners.size();
	//				corners_Seq.push_back(corners);
	//				imwrite(filename, frame);
	//				tempname.clear();
	//				filename.clear();
	//			}
	//			else
	//			{
	//				std::cout << "Detect Failed.\n";
	//			}
	//		//}
	//	}
	//	/*msg = mode == CAPTURING ? "100/100/s" : mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";*/
	//	baseLine = 0;
	//	textSize = getTextSize(msg, 1, 1, 1, &baseLine);
	//	Point textOrigin(frame.cols - 2 * textSize.width - 10, frame.rows - 2 * baseLine - 10);
	//
	//	if (mode == CAPTURING)
	//	{
	//		msg = format("%d/%d", n, image_count);
	//	}
	//
	///*	putText(frame, msg, textOrigin, 1, 1, mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));*/
	//	imshow("Calibration", frame);
	//	key = 0xff & waitKey(1);
	//	if ((key & 255) == 27)
	//		break;
	//}
	//
	//std::cout << "角点提取完成！\n";
	
	
	/************************************************************************
	离线标定方法：通过摄像机拍摄多张标定板不同姿态的图像（一般10幅以上），设置相应参数后进行标定
	*************************************************************************/

	int image_count = 20;                    /****    图像数量     ****/
	Mat frame;
	Size board_size = Size(7,5);            /****标定板上每行、列的角点数 ****///不同标定板size也是不同的，根据自己的标定板更改为对应值。
	//vector<Point2f> corners;                  /****    缓存每幅图像上检测到的角点       ****/
	vector<vector<Point2f>>  corners_Seq;    /****  保存摄像机所有照片检测到的角点       ****/
	ofstream fout("calibration_result.txt");  /**    保存定标结果的文件     **/

	Mat rgbImageL, grayImageL;
	const int imageWidth = 960;								//摄像头的分辨率
	const int imageHeight = 540;	
	vector<Point2f> corners;								//摄像机某一照片角点坐标集合
	Size image_size = Size(imageWidth, imageHeight);            /****     图像的尺寸      ****/


	Mat img;
	int count = 0;                                          //被读出图像数量
	/*namedWindow("ImageL");
	namedWindow("ImageR");*/
	cout << "按Q退出 ..." << endl;
	while (count < image_count)                                          
	{
		char filename[100];

		/*读取图像*/
		sprintf_s(filename, "image201709042\\Left%02d.bmp", count + 1);    //读取image1文件夹下的图片
		rgbImageL = imread(filename, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);

		bool isFind;

		isFind = findChessboardCorners(rgbImageL, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		if (isFind == true)	 //如果两幅图像都找到了所有的角点 则说明这两幅图像是可行的
		{
			/*
			Size(5,5) 搜索窗口的一半大小
			Size(-1,-1) 死区的一半尺寸
			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)迭代终止条件
			*/
			cornerSubPix(grayImageL, corners, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
			drawChessboardCorners(rgbImageL, board_size, corners, isFind);
			Mat rgbImageL1;
			resize(rgbImageL,rgbImageL1,Size(),0.3,0.3);
			imshow("chessboardL", rgbImageL1);
			corners_Seq.push_back(corners);

			count++;
			cout << "The image is good!" << endl;
		}
		else
		{
			cout << "The image is bad ,please try again" << endl;
		}

		if (waitKey(10) == 'q')
		{
			break;
		}
	}


	/************************************************************************
		   摄像机定标
     *************************************************************************/
	std::cout << "开始定标………………" << endl;
	Size square_size = Size(30, 30);             /**** 实际测量得到的标定板上每个棋盘格的大小,实际标定板上的棋盘格尺寸（mm）   ****/
	vector<vector<Point3f>>  object_Points;                                      /****  保存定标板上角点的三维坐标   ****/

	Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0));          /*****   保存提取的所有角点   *****/
	vector<int>  point_counts;                                          /*****    每幅图像中角点的数量    ****/
	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));                /*****    摄像机内参数矩阵    ****/
	Mat distortion_coeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));            /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	vector<Mat> rotation_vectors;                                      /* 每幅图像的旋转向量 */
	vector<Mat> translation_vectors;                                  /* 每幅图像的平移向量 */

	/* 初始化定标板上角点的三维坐标 */
	for (int t = 0; t < image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < board_size.height; i++)
		{
			for (int j = 0; j < board_size.width; j++)
			{
				/* 假设定标板放在世界坐标系中z=0的平面上 */
				Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);
	}
	
	/* 初始化每幅图像中的角点数，这里我们假设每幅图像中都可以看到完整的定标板 */
	for (int i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	/* 开始定标 */

	calibrateCamera(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors);
	std::cout << "定标完成！\n";

	/************************************************************************
		   对定标结果进行评价
		   *************************************************************************/
	std::cout << "开始评价定标结果………………" << endl;
	double total_err = 0.0;                   /* 所有图像的平均误差的总和 */
	double err = 0.0;                        /* 每幅图像的平均误差 */
	vector<Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/

	std::cout << "每幅图像的定标误差：" << endl;
	fout << "每幅图像的定标误差：" << endl << endl;
	for (int i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];
		/****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
		projectPoints(tempPointSet, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs, image_points2);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = corners_Seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		std::cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	std::cout << "总体平均误差：" << total_err / image_count << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
	std::cout << "评价完成！" << endl;

	/************************************************************************
		   保存定标结果
		   *************************************************************************/
	std::cout << "开始保存定标结果………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */

	fout << "相机内参数矩阵：" << endl;
	fout << intrinsic_matrix << endl << endl;
	fout << "畸变系数：\n";
	fout << distortion_coeffs << endl << endl << endl;
	for (int i = 0; i < image_count; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << rotation_vectors[i] << endl;

		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(rotation_vectors[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << translation_vectors[i] << endl << endl;
	}
	std::cout << "完成保存" << endl;
	fout << endl;

	/************************************************************************  
           显示定标结果  
    *************************************************************************/
 	Mat mapx = Mat(image_size,CV_32FC1);
 	Mat mapy = Mat(image_size,CV_32FC1);
 	Mat R = Mat::eye(3,3,CV_32F);
 	std::cout<<"保存矫正图像"<<endl;
 	string imageFileName;
 	std::stringstream StrStm;
 	for (int i = 0 ; i != image_count ; i++)
 	{
 		std::cout<<"Frame #"<<i+1<<"..."<<endl;
 		Mat newCameraMatrix = Mat(3,3,CV_32FC1,Scalar::all(0));
		initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, newCameraMatrix /*intrinsic_matrix*/, image_size, CV_32FC1, mapx, mapy);
 		StrStm.clear();
 		imageFileName.clear();
 		StrStm<<i+1;
 		StrStm>>imageFileName;
 		imageFileName += ".jpg";
		Mat t = imread(imageFileName);
 		Mat newimage = t.clone();
 		cv::remap(t,newimage,mapx, mapy, INTER_LINEAR);
 		StrStm.clear();
 		imageFileName.clear();
 		StrStm<<i+1;
 		StrStm>>imageFileName;
 		imageFileName += "_d.jpg";
 		imwrite(imageFileName,newimage);
 	}
 	std::cout<<"保存结束"<<endl;
	return 0;
}