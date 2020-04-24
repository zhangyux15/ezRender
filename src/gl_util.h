#pragma once
#include <map>
#include <string>
#include <iostream>
#include <queue>
#include <mutex>
#include <variant>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/eigen.hpp>
#include "model.h"


namespace GLUtil
{
	enum ShaderType
	{
		SHADER_COLOR,
		SHADER_TEXTURE,
		SHADER_TYPE_SIZE
	};

	enum UBOType
	{
		UBO_CAMERA,
		UBO_TYPE_SIZE
	};

	unsigned int GetShader(const ShaderType& type);
	unsigned int CreateShader(const std::string& vs, const std::string& gs, const std::string& fs);
	unsigned int GetUBO(const UBOType& type);
	unsigned int CreateUBO(const int& bytes);

	struct UniformVariable
	{
		UniformVariable() {}
		UniformVariable(const GLenum& _type) { Set(_type); }
		void Set(const GLenum& _type);
		void Upload(const unsigned int& shaderId, const int& location) const;
		void Download(const unsigned int& shaderId, const int& location);

		GLenum type;
		std::vector<char> data;
	};

	struct UniformBuffer
	{
		unsigned int id, index;
		UniformBuffer(const ShaderType& shaderType, const std::string& name, const int& bytes);
		~UniformBuffer() { glDeleteBuffers(1, &id); }
		UniformBuffer(const UniformBuffer& _) = delete;
		UniformBuffer& operator=(const UniformBuffer& _) = delete;
	};

	struct RenderObject
	{
		unsigned int id;
		ShaderType type;

		cv::ogl::Buffer ebo;
		std::map<std::string, cv::ogl::Buffer> vbos;
		std::map<std::string, cv::ogl::Texture2D> texs;
		std::map<std::string, UniformVariable> uniforms;

		RenderObject(const ShaderType& _type);
		~RenderObject() { glDeleteVertexArrays(1, &id); }

		RenderObject(const RenderObject& _) = delete;
		RenderObject& operator=(const RenderObject& _) = delete;

		template <typename T, int rows, int cols>
		void SetBuffer(const std::string& name, const Eigen::Matrix<T, rows, cols>& arr) {
			SetBuffer(name, cv::Mat(arr.cols(), arr.rows(), cv::traits::Type<T>::value,
				(void*)arr.data(), arr.outerStride() * sizeof(T)));
		}

		void SetBuffer(const std::string& name, cv::InputArray arr, cudaStream_t stream = NULL);
		void SetTexture(const std::string& name, cv::InputArray arr);
		void SetUniform(const std::string& name, void* data);
		void SetModel(const Model& model);

		void Draw();
	};

	struct Viewer
	{
		float fov, aspect, nearest, farthest, radius;
		Eigen::Vector3f eye, center, up, front, right;
		Eigen::Matrix4f perspective, transform;

		Viewer();
		void Perspective(const float& _fov, const float& _aspect, const float& _nearest, const float& _farthest);
		void LookAt(const Eigen::Vector3f& _eye, const Eigen::Vector3f& _center, const Eigen::Vector3f& _up);
		void SetFov(const float& _fov) { Perspective(_fov, aspect, nearest, farthest); }
		void SetAspect(const float& _aspect) { Perspective(fov, _aspect, nearest, farthest); }
		void SetNearest(const float& _nearest) { Perspective(fov, aspect, _nearest, farthest); }
		void SetFarthest(const float& _farthest) { Perspective(fov, aspect, nearest, _farthest); }
		void SetEye(const Eigen::Vector3f& _eye) { LookAt(_eye, center, up); }
		void SetCenter(const Eigen::Vector3f& _center) { LookAt(eye, _center, up); }
		void SetUp(const Eigen::Vector3f& _up) { LookAt(eye, center, _up); }
		void SetRadius(const float& _radius) { SetEye(center - front * _radius); }
		void SetCam(const Eigen::Matrix3f& K, const Eigen::Matrix3f& R, const Eigen::Vector3f& T);

		Eigen::Vector3f ArcballMap(const Eigen::Vector2f& p) const;
		void DragEye(const Eigen::Vector2f& pBegin, const Eigen::Vector2f& pEnd);
		void DragCenter(const Eigen::Vector2f& pBegin, const Eigen::Vector2f& pEnd);
		void Upload();
	};
}


