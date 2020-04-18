#pragma once
#include <Eigen/Core>

struct Model
{
	Model() {}
	Model(const std::string& filename) { Load(filename); }

	Eigen::Matrix3Xf vertices;
	Eigen::Matrix3Xf normals;
	Eigen::Matrix4Xf colors;
	Eigen::Matrix3Xi faces;
	Eigen::Matrix2Xf texcoords;

	void CalcNormal();
	void SetColor(const Eigen::Vector4f& color);
	void Drive(const Eigen::Vector3f& scale, const Eigen::Vector3f& rotation, const Eigen::Vector3f& translation);
	void Load(const std::string& filename, const unsigned int& flags = NULL);
};

