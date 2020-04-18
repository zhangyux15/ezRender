#include "model.h"
#include <Eigen/Eigen>
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


void Model::CalcNormal()
{
	normals.setZero(3, vertices.cols());
	for (int fIdx = 0; fIdx < faces.cols(); fIdx++) {
		const auto face = faces.col(fIdx);
		Eigen::Vector3f normal = ((vertices.col(face.x()) - vertices.col(face.y())).cross(
			vertices.col(face.y()) - vertices.col(face.z()))).normalized();

		normals.col(face.x()) += normal;
		normals.col(face.y()) += normal;
		normals.col(face.z()) += normal;
	}
	normals.colwise().normalize();
}

void Model::SetColor(const Eigen::Vector4f& color)
{
	colors.resize(4, vertices.cols());
	colors.colwise() = color;
}

void Model::Drive(const Eigen::Vector3f& scale, const Eigen::Vector3f& rotation, const Eigen::Vector3f& translation)
{
	const Eigen::Matrix3f scaleMat = scale.asDiagonal();
	const Eigen::Matrix3f rotMat = rotation.norm() < FLT_EPSILON ? 
		Eigen::Matrix3f::Identity() : Eigen::AngleAxisf(rotation.norm(), rotation.normalized()).matrix();
	vertices = rotMat * scaleMat * vertices;
	vertices.colwise() += translation;
	normals = rotMat * normals;
}

void Model::Load(const std::string& filename, const unsigned int& flags)
{
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(filename, flags | aiProcess_Triangulate | aiProcess_GenNormals);
	// check for errors
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cerr << "Error assimp: " << importer.GetErrorString() << std::endl;
		std::abort();
	}

	bool flag = false;
	auto ProcessMesh = [this, scene, &flag](const aiMesh * const mesh) {
		if (flag) {
			std::cerr << "more than one mesh in file" << std::endl;
			std::abort();
		}

		flag = true;
		vertices.resize(3, mesh->mNumVertices);
		for (unsigned int vIdx = 0; vIdx < mesh->mNumVertices; vIdx++)
			vertices.col(vIdx) = Eigen::Vector3f(mesh->mVertices[vIdx].x, mesh->mVertices[vIdx].y, mesh->mVertices[vIdx].z);

		normals.resize(3, mesh->mNumVertices);
		for (unsigned int vIdx = 0; vIdx < mesh->mNumVertices; vIdx++)
			normals.col(vIdx) = Eigen::Vector3f(mesh->mNormals[vIdx].x, mesh->mNormals[vIdx].y, mesh->mNormals[vIdx].z);

		if (mesh->mTextureCoords[0]) {
			texcoords.resize(2, mesh->mNumVertices);
			for (unsigned int vIdx = 0; vIdx < mesh->mNumVertices; vIdx++)
				texcoords.col(vIdx) = Eigen::Vector2f(mesh->mTextureCoords[0][vIdx].x, mesh->mTextureCoords[0][vIdx].y);
		}

		if (mesh->mColors[0]) {
			colors.resize(4, mesh->mNumVertices);
			for (unsigned int vIdx = 0; vIdx < mesh->mNumVertices; vIdx++)
				colors.col(vIdx) = Eigen::Vector4f(mesh->mColors[0][vIdx].r, mesh->mColors[0][vIdx].g,
					mesh->mColors[0][vIdx].b, mesh->mColors[0][vIdx].a);
		}

		faces.resize(3, mesh->mNumFaces);
		for (unsigned int fIdx = 0; fIdx < mesh->mNumFaces; fIdx++)
			faces.col(fIdx) = Eigen::Vector3i(mesh->mFaces[fIdx].mIndices[0],
				mesh->mFaces[fIdx].mIndices[1], mesh->mFaces[fIdx].mIndices[2]);

	};

	std::function<void(const aiNode * const, std::vector<Model>&)> ProcessNode = [scene, &ProcessMesh, &ProcessNode](const aiNode * const node, std::vector<Model>& models) {
		for (unsigned int i = 0; i < node->mNumMeshes; i++) {
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			ProcessMesh(mesh);
		}

		for (unsigned int i = 0; i < node->mNumChildren; i++)
			ProcessNode(node->mChildren[i], models);
	};

	std::vector<Model> models;
	ProcessNode(scene->mRootNode, models);
}


