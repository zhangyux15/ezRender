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
		UniformBuffer(const ShaderType& shaderType, const std::string& name, const int& bytes) {
			glGenBuffers(1, &id);
			index = glGetUniformBlockIndex(GetShader(shaderType), name.c_str());
			glBindBuffer(GL_UNIFORM_BUFFER, id);
			glBufferData(GL_UNIFORM_BUFFER, bytes, NULL, GL_STATIC_DRAW);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
		}
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


	//struct EBO
	//{
	//	unsigned int id, dim, type;
	//	size_t size = 0, bytes = 0;

	//	EBO(const unsigned int& _type) {
	//		type = _type;
	//		switch (type)
	//		{
	//		case GL_TRIANGLES:
	//			dim = 3;
	//			break;
	//		case GL_LINES:
	//			dim = 2;
	//			break;
	//		default:
	//			std::cerr << "Unsupported type" << std::endl;
	//			std::abort();
	//			break;
	//		}
	//		glGenBuffers(1, &id);
	//	}
	//	~EBO() { glDeleteBuffers(1, &id); }
	//	EBO(const EBO& _) = delete;
	//	EBO& operator=(const EBO& _) = delete;
	//	void Bind() { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id); }
	//	void Draw() { glDrawElements(type, (int)size, GL_UNSIGNED_INT, NULL); }
	//	void Upload(const unsigned int* const data, const size_t& _size, const unsigned int& type = GL_DYNAMIC_DRAW)
	//	{
	//		size = _size;
	//		bytes = size * sizeof(unsigned int);
	//		Bind();
	//		glBufferData(GL_ELEMENT_ARRAY_BUFFER, bytes, data, type);
	//	}

	//	void Upload(const Eigen::MatrixXu& data)
	//	{
	//		assert((unsigned int)data.rows() == dim);
	//		Upload(data.data(), data.size());
	//	}

	//	void Download(unsigned int* data)
	//	{
	//		Bind();
	//		glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, bytes, data);
	//	}

	//	void Download(Eigen::MatrixXu& data)
	//	{
	//		data.resize(dim, size / dim);
	//		Download(data.data());
	//	}
	//};


	//struct VBO
	//{
	//	unsigned int id, dim, type, status;
	//	size_t size = 0, bytes = 0;

	//	VBO(const unsigned int& _type, const unsigned int& _status = GL_DYNAMIC_DRAW)
	//	{
	//		type = _type;
	//		status = _status;
	//		glGenBuffers(1, &id);
	//	}

	//	~VBO() { glDeleteBuffers(1, &id); }
	//	VBO(const VBO& _) = delete;
	//	VBO& operator=(const VBO& _) = delete;

	//	void Bind() { glBindBuffer(GL_ARRAY_BUFFER, id); }

	//	template <typename T>
	//	void Upload(const T* const data, const unsigned int& _dim, const size_t& _size)
	//	{
	//		assert(Type2Enum<T>() == type);
	//		dim = _dim;
	//		size = _size;
	//		bytes = size * sizeof(T);
	//		Bind();
	//		glBufferData(GL_ARRAY_BUFFER, bytes, data, status);
	//	}

	//	template <typename T>
	//	void Upload(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data)
	//	{
	//		Upload(data.data(), (unsigned int)data.rows(), data.size());
	//	}

	//	template <typename T>
	//	void Download(T* data)
	//	{
	//		assert(Type2Enum<T>() == type);
	//		Bind();
	//		glGetBufferSubData(GL_ARRAY_BUFFER, 0, bytes, data);
	//	}

	//	template <typename T>
	//	void Download(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data)
	//	{
	//		data.resize(dim, size / dim);
	//		Download(data.data());
	//	}
	//};


	//struct VAO
	//{
	//	unsigned int id;
	//	ShaderType shaderType;

	//	std::shared_ptr<Tex> texuv;
	//	std::shared_ptr<EBO> ebo;
	//	std::map<AttribType, std::shared_ptr<VBO>> vbos;
	//	std::map<std::string, UniformVar> vars;

	//	VAO() {
	//		glGenVertexArrays(1, &id);

	//		shaderType = SHADER_AUTO;
	//		ebo = std::make_shared<EBO>(GL_TRIANGLES);
	//		Bind();
	//		ebo->Bind();

	//		vars.insert(std::make_pair("u_ambient", UniformVar(.5f)));
	//		vars.insert(std::make_pair("u_diffuse", UniformVar(.5f)));
	//		vars.insert(std::make_pair("u_specular", UniformVar(.01f)));
	//		vars.insert(std::make_pair("u_shininess", UniformVar(20.f)));
	//		vars.insert(std::make_pair("u_alpha", UniformVar(1.f)));
	//	}
	//	VAO(const Model& model) :VAO() { UploadModel(model); }
	//	~VAO() { glDeleteVertexArrays(1, &id); }
	//	VAO(const VAO& _) = delete;
	//	VAO& operator=(const VAO& _) = delete;

	//	void Bind() { glBindVertexArray(id); }

	//	template <typename T>
	//	void UploadVar(const std::string& name, const T& var) { vars[name] = UniformVar(var); }

	//	template <typename T>
	//	void UploadVBO(const AttribType& attrib, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data)
	//	{
	//		Bind();
	//		auto iter = vbos.find(attrib);
	//		if (iter == vbos.end())
	//			iter = vbos.insert(std::make_pair(attrib, std::make_shared<VBO>(Type2Enum<T>()))).first;

	//		iter->second->Upload(data.data(), (unsigned int)data.rows(), data.size());
	//		glVertexAttribPointer(attrib, iter->second->dim, iter->second->type, GL_FALSE, 0, NULL);			// already bind vbo when call upload()
	//		glEnableVertexAttribArray(attrib);
	//	}

	//	void UploadEBO(const Eigen::MatrixXu& data)
	//	{
	//		Bind();
	//		ebo->Upload(data);
	//	}

	//	void UploadTex(const cv::Mat& img, const unsigned int& format)
	//	{
	//		cv::Size _size(img.cols, img.rows);
	//		if (!texuv || texuv->size != _size)
	//			texuv = std::make_shared<Tex>(_size);
	//		texuv->Upload(img, format);
	//	}

	//	void UploadTex(const cv::cuda::GpuMat& img, cudaStream_t stream = NULL)
	//	{
	//		cv::Size _size(img.cols, img.rows);
	//		if (!texuv || texuv->size != _size)
	//			texuv = std::make_shared<Tex>(_size);
	//		texuv->Upload(img, stream);
	//	}

	//	void UploadModel(const Model& model) {
	//		UploadVBO<float>(ATTRIB_VERTEX, model.vertices);
	//		UploadVBO<float>(ATTRIB_NORMAL, model.normals);
	//		UploadEBO(model.faces);
	//		if (model.colors.size() > 0)
	//			UploadVBO<float>(ATTRIB_COLOR, model.colors);
	//		if (model.texcoords.size() > 0)
	//			UploadVBO<float>(ATTRIB_TEXCOORD, model.texcoords);
	//	}

	//	void Draw()
	//	{
	//		Bind();
	//		ShaderType _shaderType = shaderType != SHADER_AUTO ? shaderType
	//			: vbos.find(ATTRIB_TEXCOORD) != vbos.end() ? SHADER_TEXTURE
	//			: vbos.find(ATTRIB_COLOR) != vbos.end() ? SHADER_COLOR
	//			: SHADER_UNKNOWN;
	//		std::shared_ptr<Shader> shader = Shader::Get(_shaderType);

	//		// uniform var
	//		for (const auto& var : vars)
	//			var.second.Upload(shader, var.first);

	//		// uniform tex
	//		if (texuv) {
	//			glActiveTexture(GL_TEXTURE0 + TEXINDEX_UV);
	//			UniformVar(int(TEXINDEX_UV)).Upload(shader, TEX_UNIFORM[TEXINDEX_UV]);
	//			texuv->Bind();
	//		}

	//		ebo->Draw();
	//	}
	//};

}


