#include <iostream>
#include <sstream>
#include <Eigen/Eigen>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "camera.h"
#include "math_util.h"


void Camera::Parse(const Json::Value& json)
{
	Json::Value var = json["K"];
	for (int row = 0; row < 3; row++)
		for (int col = 0; col < 3; col++)
			cvK(row, col) = var[row * 3 + col].asFloat();

	if (json.isMember("R")) {
		var = json["R"];
		if (var.size() == 3) {
			cv::Vec3f r;
			for (int i = 0; i < 3; i++)
				r[i] = var[i].asFloat();
			cv::Rodrigues(r, cvR);
		}
		else if (var.size() == 9) {
			for (int row = 0; row < 3; row++)
				for (int col = 0; col < 3; col++)
					cvR(row, col) = var[row * 3 + col].asFloat();
		}
		else {
			std::cerr << "Unknown rotation format" << std::endl;
		}
	}

	if (json.isMember("T")) {
		var = json["T"];
		for (int i = 0; i < 3; i++)
			cvT[i] = var[i].asFloat();
	}

	if (json.isMember("RT")) {
		var = json["RT"];
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 3; col++)
				cvR(row, col) = var[row * 4 + col].asFloat();
			cvT[row] = var[row * 4 + 3].asFloat();
		}
	}

	var = json["imgSize"];
	cvImgSize = cv::Size(var[0].asFloat(), var[1].asFloat());

	if (json.isMember("distCoeff")) {
		var = json["distCoeff"];
		for (int i = 0; i < int(cvDistCoeff.rows); i++)
			cvDistCoeff(i) = var[i].asFloat();
	}

	if (json.isMember("newImgSize")) {
		var = json["newImgSize"];
		cvNewImgSize = cv::Size(var[0].asFloat(), var[1].asFloat());
	}

	if (json.isMember("rectifyAlpha")) 
		cvRectifyAlpha = json["rectifyAlpha"].asDouble();

	Update();
}


void Camera::Update()
{
	cvNewImgSize = cvNewImgSize == cv::Size(0, 0) ? cvImgSize : cvNewImgSize;

	cvNewK = cv::getOptimalNewCameraMatrix(
		cvK, cvDistCoeff, cvImgSize, cvRectifyAlpha, cvNewImgSize);
	cv::initUndistortRectifyMap(
		cvK, cvDistCoeff, cv::Mat(), cvNewK, cvNewImgSize, CV_32FC1, cvRectifyMapX, cvRectifyMapY);

	cv::cv2eigen(cvNewK, K);
	K.row(0) /= float(cvNewImgSize.width);
	K.row(1) /= float(cvNewImgSize.height);

	cv::cv2eigen(cvR, R);
	cv::cv2eigen(cvT, T);
	Ki = K.inverse();
	Rt = R.transpose();
	RtKi = Rt * Ki;
	pos = -Rt * T;
	proj.leftCols<3>() = R;
	proj.col(3) = T;
	proj = K * proj;
}


void Camera::LookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up)
{
	R.row(2) = (center - eye).transpose().normalized();
	R.row(0) = (up.cross(R.row(2).transpose())).transpose().normalized();
	R.row(1) = (R.row(2).transpose().cross(R.row(0).transpose())).transpose().normalized();
	T = -R * eye;
	cv::eigen2cv(R, cvR);
	cv::eigen2cv(T, cvT);
	Update();
}


Json::Value Camera::Serialize() const
{
	Json::Value json;
	json["K"].resize(0);
	for (int row = 0; row < 3; row++)
		for (int col = 0; col < 3; col++)
			json["K"].append(Json::Value(cvK(row, col)));

	json["R"].resize(0);
	for (int row = 0; row < 3; row++)
		for (int col = 0; col < 3; col++)
			json["R"].append(Json::Value(cvR(row, col)));

	json["T"].resize(0);
	for (int i = 0; i < 3; i++)
		json["T"].append(Json::Value(cvT(i)));

	json["imgSize"].resize(0);
	json["imgSize"].append(Json::Value(cvImgSize.width));
	json["imgSize"].append(Json::Value(cvImgSize.height));


	json["distCoeff"].resize(0);
	for (int i = 0; i < cvDistCoeff.rows; i++)
		json["distCoeff"].append(Json::Value(cvDistCoeff(i)));

	json["newImgSize"].resize(0);
	json["newImgSize"].append(Json::Value(cvNewImgSize.width));
	json["newImgSize"].append(Json::Value(cvNewImgSize.height));

	json["rectifyAlpha"]=Json::Value(cvRectifyAlpha);
	return json;
}


Eigen::Matrix3f Camera::CalcFundamental(const Camera& camera) const
{
	const Eigen::Matrix3f relaR = R * camera.Rt;
	const Eigen::Vector3f relaT = T - relaR * camera.T;
	return (Ki.transpose())*MathUtil::Skew(relaT)*relaR*(camera.Ki);
}


Eigen::Vector3f Camera::CalcRay(const Eigen::Vector2f& uv) const
{
	return (-RtKi * uv.homogeneous()).normalized();
}


std::map<std::string, Camera> ParseCameras(const std::string& jsonFile)
{
	Json::Value json;
	std::ifstream fs(jsonFile);
	if (!fs.is_open()) {
		std::cerr << "json file not exist: " << jsonFile << std::endl;
		std::abort();
	}

	std::string errs;
	Json::parseFromStream(Json::CharReaderBuilder(), fs, &json, &errs);
	fs.close();

	if (errs != "") {
		std::cerr << "json read file error: " << errs << std::endl;
		std::abort();
	}

	std::map<std::string, Camera> cameras;
	for (auto camIter = json.begin(); camIter != json.end(); camIter++) 
		cameras.insert(std::make_pair(camIter.key().asString(), Camera(*camIter)));
	return cameras;
}


void SerializeCameras(const std::map<std::string, Camera>& cameras, const std::string& jsonFile)
{
	Json::Value json;
	for (const auto& camIter : cameras) 
		json[camIter.first] = camIter.second.Serialize();

	std::ofstream ofs(jsonFile);
	std::unique_ptr<Json::StreamWriter> sw_t(Json::StreamWriterBuilder().newStreamWriter());
	sw_t->write(json, &ofs);
	ofs.close();
}


void Triangulator::Solve(const int& maxIterTime, const float& updateTolerance, const float& regularTerm) {
	if (projs.size() < 2) 
		return;
	for (int iterTime = 0; iterTime < maxIterTime && !convergent; iterTime++) {
		Eigen::MatrixXf jacobi = Eigen::MatrixXf::Zero(2 * projs.size(), 3);
		Eigen::VectorXf residual(2 * projs.size());
		for (int i = 0; i < projs.size(); i++) {
			const Eigen::Vector3f xyz = projs[i] * pos.homogeneous();
			Eigen::Matrix<float, 2, 3> tmp;
			tmp << 1.0f / xyz.z(), 0.0f, -xyz.x() / (xyz.z()*xyz.z()),
				0.0f, 1.0f / xyz.z(), -xyz.y() / (xyz.z()*xyz.z());
			jacobi.block<2, 3>(2 * i, 0) = tmp * projs[i].block<3, 3>(0, 0);
			residual.segment<2>(2 * i) = xyz.hnormalized() - points[i];
		}

		loss = residual.cwiseAbs().sum() / residual.rows();
		Eigen::MatrixXf hessian = jacobi.transpose()*jacobi + regularTerm * Eigen::MatrixXf::Identity(jacobi.cols(), jacobi.cols());
		Eigen::VectorXf gradient = jacobi.transpose()*residual;

		const Eigen::VectorXf deltaPos = hessian.ldlt().solve(-gradient);
		if (deltaPos.norm() < updateTolerance)
			convergent = true;
		else
			pos += deltaPos;
	}
}
