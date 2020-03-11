#pragma once
#include <map>
#include <string>
#include <iostream>
#include <queue>
#include <mutex>
#include <variant>
#include <typeinfo>
#include <type_traits>
#include <thread>
#include <chrono>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <cuda_gl_interop.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "math_util.h"
#include "cuda_util.h"


namespace GLUtil
{
	template <typename T>
	unsigned int Type2Enum()
	{
		typedef std::decay<T>::type U;
		if (std::is_same<float, U>::value)
			return GL_FLOAT;
		else if (std::is_same<double, U>::value)
			return GL_DOUBLE;
		else if (std::is_same<int, U>::value)
			return GL_INT;
		else if (std::is_same<unsigned int, U>::value)
			return GL_UNSIGNED_INT;
		else if (std::is_same<char, U>::value)
			return GL_BYTE;
		else if (std::is_same<unsigned char, U>::value)
			return GL_UNSIGNED_BYTE;
		else {
			std::cerr << "Unsupported type: " << typeid(U).name() << std::endl;
			std::abort();
		}
	}

	enum ShaderType
	{
		SHADER_AUTO = -1,
		SHADER_COLOR,
		SHADER_TEXTURE,
		SHADER_UNKNOWN
	};


	enum AttribType
	{
		ATTRIB_VERTEX,
		ATTRIB_NORMAL,
		ATTRIB_COLOR,
		ATTRIB_TEXCOORD
	};


	enum TexIndex
	{
		TEXINDEX_UV,
		TEXINDEX_LIGHT_DEPTH
	};


	const std::string TEX_UNIFORM[] = {
		"u_tex_uv",
		"u_tex_light_depth"
	};


	struct Shader
	{
		static std::shared_ptr<Shader> Get(const ShaderType& type);

		unsigned int id;
		Shader(const std::string& vs, const std::string& gs, const std::string& fs)
		{
			auto CheckShaderError = [](const unsigned int& id, const std::string& type) {
				int success;
				char infoLog[1024];
				if (type != "PROGRAM")
					glGetShaderiv(id, GL_COMPILE_STATUS, &success);
				else
					glGetProgramiv(id, GL_LINK_STATUS, &success);

				if (!success) {
					glGetShaderInfoLog(id, 1024, NULL, infoLog);
					std::cerr << "Init shader Error: " << type << " " << std::string(infoLog) << std::endl;
					std::abort();
				}
			};

			id = glCreateProgram();

			unsigned int vsId = glCreateShader(GL_VERTEX_SHADER);
			const char* const vsCStr = vs.c_str();
			glShaderSource(vsId, 1, &vsCStr, NULL);
			glCompileShader(vsId);
			CheckShaderError(vsId, "VERTEX");
			glAttachShader(id, vsId);

			unsigned int fsId = glCreateShader(GL_FRAGMENT_SHADER);
			const char* const fsCStr = fs.c_str();
			glShaderSource(fsId, 1, &fsCStr, NULL);
			glCompileShader(fsId);
			CheckShaderError(fsId, "FRAGMENT");
			glAttachShader(id, fsId);

			unsigned int gsId;
			bool gsValid = gs != "";
			if (gsValid) {
				gsId = glCreateShader(GL_GEOMETRY_SHADER);
				const char* const gsCStr = gs.c_str();
				glShaderSource(gsId, 1, &gsCStr, NULL);
				glCompileShader(gsId);
				CheckShaderError(gsId, "GEOMETRY");
				glAttachShader(id, gsId);
			}
			glLinkProgram(id);
			CheckShaderError(id, "PROGRAM");
			glDeleteShader(vsId);
			glDeleteShader(fsId);
			if (gsValid)
				glDeleteShader(gsId);
		}

		~Shader() { glDeleteProgram(id); }
		Shader(const Shader& _) = delete;
		Shader& operator=(const Shader& _) = delete;
		void Use() { glUseProgram(id); }
	};


	struct UniformVar
	{
		std::variant<float, unsigned int, int,
			Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector4f,
			Eigen::Matrix2f, Eigen::Matrix3f, Eigen::Matrix4f> var;

		template <typename T>
		UniformVar(const T& _var) { var = _var; }

		struct Visitor {
			int index;
			Visitor(const int& _index) { index = _index; }
			void operator()(const float& var) { glUniform1f(index, var); }
			void operator()(const unsigned int& var) { glUniform1i(index, int(var)); }
			void operator()(const int& var) { glUniform1i(index, var); }
			void operator()(const Eigen::Vector2f& var) { glUniform2f(index, var.x(), var.y()); }
			void operator()(const Eigen::Vector3f& var) { glUniform3f(index, var.x(), var.y(), var.z()); }
			void operator()(const Eigen::Vector4f& var) { glUniform4f(index, var.x(), var.y(), var.z(), var.w()); }
			void operator()(const Eigen::Matrix2f& var) { glUniformMatrix2fv(index, 1, GL_FALSE, var.data()); }
			void operator()(const Eigen::Matrix3f& var) { glUniformMatrix3fv(index, 1, GL_FALSE, var.data()); }
			void operator()(const Eigen::Matrix4f& var) { glUniformMatrix4fv(index, 1, GL_FALSE, var.data()); }
		};

		bool Upload(std::shared_ptr<Shader> shader, const std::string& name, const int& offset = 0) const {
			shader->Use();
			int index = glGetUniformLocation(shader->id, name.c_str());
			if (index == -1)
				return false;
			std::visit(Visitor{ index + offset }, var);
			return true;
		}
	};


	struct Model
	{
		Model() {}
		Model(const std::string& filename) { Load(filename); }

		Eigen::Matrix3Xf vertices;
		Eigen::Matrix3Xf normals;
		Eigen::Matrix4Xf colors;
		Eigen::Matrix3Xu faces;
		Eigen::Matrix2Xf texcoords;

		void CalcNormal()
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

		void SetColor(const Eigen::Vector4f& color)
		{
			colors.resize(4, vertices.cols());
			colors.colwise() = color;
		}

		void Drive(const Eigen::Vector3f& scale, const Eigen::Vector3f& rotation, const Eigen::Vector3f& translation)
		{
			const Eigen::Matrix3f scaleMat = scale.asDiagonal();
			const Eigen::Matrix3f rotMat = MathUtil::Rodrigues(rotation);
			vertices = rotMat * scaleMat * vertices;
			vertices.colwise() += translation;
			normals = rotMat * normals;
		}

		void Load(const std::string& filename, const unsigned int& flags = NULL)
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
					faces.col(fIdx) = Eigen::Vector3u(mesh->mFaces[fIdx].mIndices[0],
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
	};


	struct PBO
	{
		unsigned int id;
		cv::Size size;
		size_t bytes = 0;
		cudaGraphicsResource_t buffer = nullptr;

		PBO(const cv::Size& _size)
		{
			size = _size;
			bytes = sizeof(unsigned char) * size.width * size.height * 3;
			glGenBuffers(1, &id);
			Bind();
			glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, NULL, GL_DYNAMIC_COPY);
			Unbind();
			CudaUtil::Check(cudaGraphicsGLRegisterBuffer(&buffer, id, cudaGraphicsMapFlagsNone));
		}

		~PBO()
		{
			cudaGraphicsUnregisterResource(buffer);
			glDeleteBuffers(1, &id);
		}

		PBO(const PBO& _) = delete;
		PBO& operator=(const PBO& _) = delete;
		void Bind() { glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id); }
		void Unbind() { glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); }

		void Map(unsigned char** data, cudaStream_t stream)
		{
			size_t size;
			CudaUtil::Check(cudaGraphicsMapResources(1, &buffer, stream));
			CudaUtil::Check(cudaGraphicsResourceGetMappedPointer((void**)data, &size, buffer));
		}

		void Unmap(cudaStream_t stream) { CudaUtil::Check(cudaGraphicsUnmapResources(1, &buffer, stream)); }
	};


	struct Tex
	{
		unsigned int id;
		cv::Size size;
		cudaGraphicsResource_t buffer = nullptr;

		Tex(const cv::Size& _size)
		{
			size = _size;
			glGenTextures(1, &id);
			Bind();
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.width, size.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			CudaUtil::Check(cudaGraphicsGLRegisterImage(&buffer, id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
		}

		~Tex()
		{
			cudaGraphicsUnregisterResource(buffer);
			glDeleteTextures(1, &id);
		}

		Tex(const Tex& _) = delete;
		Tex& operator=(const Tex& _) = delete;
		void Bind() { glBindTexture(GL_TEXTURE_2D, id); }

		void Upload(const unsigned char* const data, const unsigned int& format)
		{
			Bind();
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.width, size.height, format, GL_UNSIGNED_BYTE, data);
		}

		void Upload(PBO& pbo, const unsigned int& format)
		{
			pbo.Bind();
			Bind();
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.width, size.height, format, GL_UNSIGNED_BYTE, NULL);
			pbo.Unbind();
		}

		void Clear()
		{
			std::vector<unsigned char> tmp(3 * size.width*size.height, 0);
			Upload(tmp.data(), GL_RGB);
		}

		void Upload(const cv::Mat& mat, const unsigned int& format)
		{
			assert(size.width == mat.cols && size.height == mat.rows);
			Upload(mat.data, format);
		}

		void Upload(const cv::cuda::GpuMat& mat, cudaStream_t stream)
		{
			// only support 4 channel
			assert(mat.type() == CV_8UC4 && size.width == mat.cols&&size.height == mat.rows);

			cudaArray_t data_t;
			Map(&data_t, stream);
			CudaUtil::Check(cudaMemcpy2DToArrayAsync(data_t, 0, 0, mat.data, mat.step, mat.cols * mat.elemSize(), mat.rows, cudaMemcpyDeviceToDevice, stream));
			Unmap(stream);
		}

		void Map(cudaArray_t* data, cudaStream_t stream)
		{
			CudaUtil::Check(cudaGraphicsMapResources(1, &buffer, stream));
			CudaUtil::Check(cudaGraphicsSubResourceGetMappedArray(data, buffer, 0, 0));
		}

		void Unmap(cudaStream_t stream) { CudaUtil::Check(cudaGraphicsUnmapResources(1, &buffer, stream)); }
	};


	struct RBO
	{
		unsigned int id;
		unsigned int type;
		cv::Size size;

		RBO(const unsigned int& _type, const cv::Size& _size) {
			type = _type;
			size = _size;
			glGenRenderbuffers(1, &id);
			Bind();
			glRenderbufferStorage(GL_RENDERBUFFER, type, size.width, size.height);
			Unbind();
		}

		~RBO() { glDeleteRenderbuffers(1, &id); }

		RBO(const RBO& _) = delete;
		RBO& operator=(const RBO& _) = delete;
		void Bind() { glBindRenderbuffer(GL_FRAMEBUFFER, id); }
		void Unbind() { glBindRenderbuffer(GL_FRAMEBUFFER, 0); }
	};


	struct FBO
	{
		unsigned int id;
		cv::Size size;
		RBO colorRBO, depthRBO;

		FBO(const cv::Size& _size)
			: colorRBO(GL_RGBA8, _size), depthRBO(GL_DEPTH24_STENCIL8, _size) {
			size = _size;
			glGenFramebuffers(1, &id);
			Bind();
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorRBO.id);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO.id);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthRBO.id);
			glDrawBuffer(GL_COLOR_ATTACHMENT0);
			glReadBuffer(GL_COLOR_ATTACHMENT0);
			if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
				std::cerr << "FBO status not complete" << std::endl;
				std::abort();
			}
		}

		~FBO() { glDeleteFramebuffers(1, &id); }
		FBO(const FBO& _) = delete;
		FBO& operator=(const FBO& _) = delete;
		void Bind() { glBindFramebuffer(GL_FRAMEBUFFER, id); }
		void Unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
	};


	struct UBO
	{
		unsigned int id, bindPoint;

		UBO(const int& _bindPoint) {
			bindPoint = _bindPoint;
			glGenBuffers(1, &id);
			glBindBufferBase(GL_UNIFORM_BUFFER, bindPoint, id);
		}
		~UBO() { glDeleteBuffers(1, &id); }
		UBO(const UBO& _) = delete;
		UBO& operator=(const UBO& _) = delete;
		void Bind() { glBindBuffer(GL_UNIFORM_BUFFER, id); }
		void Unbind() { glBindBuffer(GL_UNIFORM_BUFFER, NULL); }
		bool Bind2Shader(std::shared_ptr<Shader> shader, const std::string& name) {
			const unsigned int index = glGetUniformBlockIndex(shader->id, name.c_str());
			if (index != GL_INVALID_INDEX) {
				glUniformBlockBinding(shader->id, index, bindPoint);
				return true;
			}
			else
				return false;
		}

		template <typename T>
		void Upload(const T& data, const unsigned int& type = GL_DYNAMIC_DRAW)
		{
			Bind();
			glBufferData(GL_UNIFORM_BUFFER, sizeof(T), (const unsigned char* const)&data, type);
			Unbind();
		}


		template <typename T>
		void Download(T& data)
		{
			Bind();
			glGetBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(T), (unsigned char*)&data);
			Unbind();
		}

		template <typename T>
		T* Map()
		{
			Bind();
			glMapBufferRange(GL_UNIFORM_BUFFER, 0, sizeof(T), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
		}

		void Unmap()
		{
			glUnmapBuffer(GL_UNIFORM_BUFFER);
			Unbind();
		}
	};


	struct EBO
	{
		unsigned int id, dim, type;
		size_t size = 0, bytes = 0;

		EBO(const unsigned int& _type) {
			type = _type;
			switch (type)
			{
			case GL_TRIANGLES:
				dim = 3;
				break;
			case GL_LINES:
				dim = 2;
				break;
			default:
				std::cerr << "Unsupported type" << std::endl;
				std::abort();
				break;
			}
			glGenBuffers(1, &id);
		}
		~EBO() { glDeleteBuffers(1, &id); }
		EBO(const EBO& _) = delete;
		EBO& operator=(const EBO& _) = delete;
		void Bind() { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id); }
		void Draw() { glDrawElements(type, (int)size, GL_UNSIGNED_INT, NULL); }
		void Upload(const unsigned int* const data, const size_t& _size, const unsigned int& type = GL_DYNAMIC_DRAW)
		{
			size = _size;
			bytes = size * sizeof(unsigned int);
			Bind();
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, bytes, data, type);
		}

		void Upload(const Eigen::MatrixXu& data)
		{
			assert((unsigned int)data.rows() == dim);
			Upload(data.data(), data.size());
		}

		void Download(unsigned int* data)
		{
			Bind();
			glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, bytes, data);
		}

		void Download(Eigen::MatrixXu& data)
		{
			data.resize(dim, size / dim);
			Download(data.data());
		}
	};


	struct VBO
	{
		unsigned int id, dim, type, status;
		size_t size = 0, bytes = 0;

		VBO(const unsigned int& _type, const unsigned int& _status = GL_DYNAMIC_DRAW)
		{
			type = _type;
			status = _status;
			glGenBuffers(1, &id);
		}

		~VBO() { glDeleteBuffers(1, &id); }
		VBO(const VBO& _) = delete;
		VBO& operator=(const VBO& _) = delete;

		void Bind() { glBindBuffer(GL_ARRAY_BUFFER, id); }

		template <typename T>
		void Upload(const T* const data, const unsigned int& _dim, const size_t& _size)
		{
			assert(Type2Enum<T>() == type);
			dim = _dim;
			size = _size;
			bytes = size * sizeof(T);
			Bind();
			glBufferData(GL_ARRAY_BUFFER, bytes, data, status);
		}

		template <typename T>
		void Upload(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data)
		{
			Upload(data.data(), (unsigned int)data.rows(), data.size());
		}

		template <typename T>
		void Download(T* data)
		{
			assert(Type2Enum<T>() == type);
			Bind();
			glGetBufferSubData(GL_ARRAY_BUFFER, 0, bytes, data);
		}

		template <typename T>
		void Download(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data)
		{
			data.resize(dim, size / dim);
			Download(data.data());
		}
	};


	struct VAO
	{
		unsigned int id;
		ShaderType shaderType;

		std::shared_ptr<Tex> texuv;
		std::shared_ptr<EBO> ebo;
		std::map<AttribType, std::shared_ptr<VBO>> vbos;
		std::map<std::string, UniformVar> vars;

		VAO() {
			glGenVertexArrays(1, &id);

			shaderType = SHADER_AUTO;
			ebo = std::make_shared<EBO>(GL_TRIANGLES);
			Bind();
			ebo->Bind();

			vars.insert(std::make_pair("u_ambient", UniformVar(.5f)));
			vars.insert(std::make_pair("u_diffuse", UniformVar(.5f)));
			vars.insert(std::make_pair("u_specular", UniformVar(.01f)));
			vars.insert(std::make_pair("u_shininess", UniformVar(20.f)));
			vars.insert(std::make_pair("u_alpha", UniformVar(1.f)));
		}

		~VAO() { glDeleteVertexArrays(1, &id); }
		VAO(const VAO& _) = delete;
		VAO& operator=(const VAO& _) = delete;

		void Bind() { glBindVertexArray(id); }

		template <typename T>
		void UploadVar(const std::string& name, const T& var) { vars[name] = UniformVar(var); }

		template <typename T>
		void UploadVBO(const AttribType& attrib, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data)
		{
			Bind();
			auto iter = vbos.find(attrib);
			if (iter == vbos.end())
				iter = vbos.insert(std::make_pair(attrib, std::make_shared<VBO>(Type2Enum<T>()))).first;

			iter->second->Upload(data.data(), (unsigned int)data.rows(), data.size());
			glVertexAttribPointer(attrib, iter->second->dim, iter->second->type, GL_FALSE, 0, NULL);			// already bind vbo when call upload()
			glEnableVertexAttribArray(attrib);
		}

		void UploadEBO(const Eigen::MatrixXu& data)
		{
			Bind();
			ebo->Upload(data);
		}

		void UploadTex(const cv::Mat& img, const unsigned int& format)
		{
			cv::Size _size(img.cols, img.rows);
			if (!texuv || texuv->size != _size)
				texuv = std::make_shared<Tex>(_size);
			texuv->Upload(img, format);
		}

		void UploadTex(const cv::cuda::GpuMat& img, cudaStream_t stream = NULL)
		{
			cv::Size _size(img.cols, img.rows);
			if (!texuv || texuv->size != _size)
				texuv = std::make_shared<Tex>(_size);
			texuv->Upload(img, stream);
		}

		void UploadModel(const Model& model) {
			UploadVBO<float>(ATTRIB_VERTEX, model.vertices);
			UploadVBO<float>(ATTRIB_NORMAL, model.normals);
			UploadEBO(model.faces);
			if (model.colors.size() > 0)
				UploadVBO<float>(ATTRIB_COLOR, model.colors);
			if (model.texcoords.size() > 0)
				UploadVBO<float>(ATTRIB_TEXCOORD, model.texcoords);
		}

		void Draw()
		{
			Bind();
			ShaderType _shaderType = shaderType != SHADER_AUTO ? shaderType
				: vbos.find(ATTRIB_TEXCOORD) != vbos.end() ? SHADER_TEXTURE
				: vbos.find(ATTRIB_COLOR) != vbos.end() ? SHADER_COLOR
				: SHADER_UNKNOWN;
			std::shared_ptr<Shader> shader = Shader::Get(_shaderType);

			// uniform var
			for (const auto& var : vars)
				var.second.Upload(shader, var.first);

			// uniform tex
			if (texuv) {
				glActiveTexture(GL_TEXTURE0 + TEXINDEX_UV);
				UniformVar(int(TEXINDEX_UV)).Upload(shader, TEX_UNIFORM[TEXINDEX_UV]);
				texuv->Bind();
			}

			ebo->Draw();
		}
	};


	struct Viewer
	{
		float fov, aspect, nearest, farthest, radius;
		Eigen::Vector3f eye, center, up, front, right;
		Eigen::Matrix4f perspective, transform;

		Viewer()
		{
			Perspective(float(EIGEN_PI) / 4.f, 1.f, 1e-2f, 1e4f);
			LookAt(Eigen::Vector3f(0.f, 0.f, 1.f), Eigen::Vector3f::Zero(), Eigen::Vector3f(0.f, -1.f, 0.f));
		}

		void Perspective(const float& _fov, const float& _aspect, const float& _nearest, const float& _farthest)
		{
			fov = _fov;
			aspect = _aspect;
			nearest = _nearest;
			farthest = _farthest;
			const float tangent = std::tan(0.5f * fov);
			perspective <<
				1.f / (tangent*aspect), 0.f, 0.f, 0.f,
				0.f, 1.f / tangent, 0.f, 0.f,
				0.f, 0.f, -(nearest + farthest) / (farthest - nearest), -2.f*farthest*nearest / (farthest - nearest),
				0.f, 0.f, -1.f, 0.f;
		}

		void SetFov(const float& _fov) { Perspective(_fov, aspect, nearest, farthest); }
		void SetAspect(const float& _aspect) { Perspective(fov, _aspect, nearest, farthest); }
		void SetNearest(const float& _nearest) { Perspective(fov, aspect, _nearest, farthest); }
		void SetFarthest(const float& _farthest) { Perspective(fov, aspect, nearest, _farthest); }

		void LookAt(const Eigen::Vector3f& _eye, const Eigen::Vector3f& _center, const Eigen::Vector3f& _up) {
			eye = _eye;
			center = _center;
			radius = (center - eye).norm();
			front = (center - eye).normalized();
			right = _up.cross(front).normalized();
			up = front.cross(right).normalized();
			transform.setIdentity();
			transform.topLeftCorner(3, 3) << right.transpose(), -up.transpose(), -front.transpose();
			transform.topRightCorner(3, 1) = -transform.topLeftCorner(3, 3) * eye;
		}

		void SetEye(const Eigen::Vector3f& _eye) { LookAt(_eye, center, up); }
		void SetCenter(const Eigen::Vector3f& _center) { LookAt(eye, _center, up); }
		void SetUp(const Eigen::Vector3f& _up) { LookAt(eye, center, _up); }
		void SetRadius(const float& _radius) { SetEye(center - front * _radius); }
		void SetCam(const cv::Size& imgSize, const Eigen::Matrix3f& K, const Eigen::Matrix3f& R, const Eigen::Vector3f& T) {
			aspect = float(imgSize.width) / float(imgSize.height);
			fov = 2 * std::atan(0.5f * float(imgSize.width) / K(1, 1));
			Perspective(fov, aspect, nearest, farthest);
			perspective(0, 2) = (1.f - 2.f*K(0, 2) / float(imgSize.width)) / aspect;
			perspective(1, 2) = 2.f*K(1, 2) / float(imgSize.height) - 1.f;
			eye = -R.transpose() * T;
			center = eye + radius * R.row(2).transpose();
			LookAt(eye, center, R.row(1).transpose());
		}

		Eigen::Vector3f ArcballMap(const Eigen::Vector2f& p) const {
			const Eigen::Vector2f pClip(2.f*p.x() - 1.f, 2.f*p.y() - 1.f);
			return pClip.squaredNorm() < 0.5f ? (Eigen::Vector3f(pClip.x(), pClip.y(), sqrtf(1.f - pClip.squaredNorm())))
				: (Eigen::Vector3f(pClip.x(), pClip.y(), 0.5f / pClip.norm())).normalized();
		};

		void DragEye(const Eigen::Vector2f& pBegin, const Eigen::Vector2f& pEnd)
		{
			const Eigen::Vector3f posBegin = ArcballMap(pBegin);
			const Eigen::Vector3f posEnd = ArcballMap(pEnd);
			const float angle = acos(posBegin.dot(posEnd));
			Eigen::Matrix3f R;
			R << right.transpose(), up.transpose(), front.transpose();
			const Eigen::Vector3f axis = R.transpose() * posBegin.cross(posEnd).normalized();
			Eigen::Matrix3f rot = MathUtil::Rodrigues<float>(angle* axis);
			SetEye(rot * (eye - center) + center);
		}

		void DragCenter(const Eigen::Vector2f& pBegin, const Eigen::Vector2f& pEnd)
		{
			const Eigen::Vector3f posBegin = ArcballMap(pBegin);
			const Eigen::Vector3f posEnd = ArcballMap(pEnd);
			const float angle = acos(posBegin.dot(posEnd));
			Eigen::Matrix3f R;
			R << right.transpose(), up.transpose(), front.transpose();
			const Eigen::Vector3f axis = R.transpose() * posBegin.cross(posEnd).normalized();
			Eigen::Matrix3f rot = MathUtil::Rodrigues<float>(angle* axis);
			SetCenter(rot.transpose() *(center - eye) + eye);
		}
	};


	class Taskque
	{
	public:
		Taskque() = default;
		~Taskque() { Clear(); }
		Taskque(const Taskque& _) = delete;
		Taskque& operator=(const Taskque& _) = delete;

		void Push(const std::function<void()>& task)
		{
			std::unique_lock<std::mutex> locker(m_mu);
			m_que.push(task);
			locker.unlock();
		}

		void Handle()
		{
			std::unique_lock<std::mutex> locker(m_mu);
			while (!m_que.empty()) {
				m_que.front()();
				m_que.pop();
			}
			locker.unlock();
		}

		void Clear()
		{
			std::unique_lock<std::mutex> locker(m_mu);
			while (!m_que.empty())
				m_que.pop();
			locker.unlock();
		}

	private:
		std::queue<std::function<void()>> m_que;
		std::mutex m_mu;
	};


	struct FPSGauge
	{
		float fps;
		int count;
		std::chrono::time_point<std::chrono::steady_clock> stamp;
		float elapsedThresh = 0.1f;  // s

		FPSGauge()
		{
			fps = 0.f;
			count = 0;
			stamp = std::chrono::steady_clock::now();
		}

		void Record()
		{
			count++;
			const float elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(
				std::chrono::steady_clock::now() - stamp).count();
			if (elapsed > elapsedThresh) {
				fps = float(count) / float(elapsed);
				count = 0;
				stamp = std::chrono::steady_clock::now();
			}
		}
	};
}


