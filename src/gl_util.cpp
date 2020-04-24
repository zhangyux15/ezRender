#include "gl_util.h"
#include <Eigen/Eigen>


namespace GLUtil
{
	void UniformVariable::Set(const GLenum& _type)
	{
		type = _type;
		const size_t size = [type = type] {
			switch (type)
			{
			case GL_FLOAT: return sizeof(float);
			case GL_INT: case GL_SAMPLER_2D: return sizeof(int);
			case GL_FLOAT_VEC2: return 2 * sizeof(float);
			case GL_FLOAT_VEC3: return 3 * sizeof(float);
			case GL_FLOAT_VEC4: return 4 * sizeof(float);
			case GL_FLOAT_MAT2: return 4 * sizeof(float);
			case GL_FLOAT_MAT3: return 9 * sizeof(float);
			case GL_FLOAT_MAT4: return 16 * sizeof(float);
			case GL_INT_VEC2: return 2 * sizeof(int);
			case GL_INT_VEC3: return 3 * sizeof(int);
			case GL_INT_VEC4: return 4 * sizeof(int);
			default:
				std::cerr << "Unsupport Uniform Type" << std::endl;
				std::abort();
			}
		}();
		data = std::vector<char>(size, 0);
	}

	void UniformVariable::Upload(const unsigned int& shaderId, const int& location)  const {
		if (location >= 0) {
			glUseProgram(shaderId);
			switch (type)
			{
			case GL_FLOAT: case GL_FLOAT_VEC2: case GL_FLOAT_VEC3: case GL_FLOAT_VEC4:
				glUniform1fv(location, data.size() / sizeof(float), (float*)data.data()); break;
			case GL_FLOAT_MAT2: glUniformMatrix2fv(location, 1, FALSE, (float*)data.data()); break;
			case GL_FLOAT_MAT3: glUniformMatrix3fv(location, 1, FALSE, (float*)data.data()); break;
			case GL_FLOAT_MAT4: glUniformMatrix4fv(location, 1, FALSE, (float*)data.data()); break;
			case GL_INT: case GL_SAMPLER_2D: case GL_INT_VEC2: case GL_INT_VEC3: case GL_INT_VEC4:
				glUniform1iv(location, data.size() / sizeof(int), (int*)data.data()); break;
			default:
				std::cerr << "Unsupport Uniform Type" << std::endl;
				std::abort();
			}
		}
	}

	void UniformVariable::Download(const unsigned int& shaderId, const int& location){
		if (location >= 0) {
			switch (type)
			{
			case GL_FLOAT: case GL_FLOAT_VEC2: case GL_FLOAT_VEC3: case GL_FLOAT_VEC4:
			case GL_FLOAT_MAT2: case GL_FLOAT_MAT3: case GL_FLOAT_MAT4:
				glGetUniformfv(shaderId, location, (float*)data.data()); break;
			case GL_INT: case GL_SAMPLER_2D: case GL_INT_VEC2: case GL_INT_VEC3: case GL_INT_VEC4:
				glUniform1iv(shaderId, location, (int*)data.data()); break;
			default:
				std::cerr << "Unsupport Uniform Type" << std::endl;
				std::abort();
			}
		}
	}

	UniformBuffer::UniformBuffer(const ShaderType& shaderType, const std::string& name, const int& bytes) {
		glGenBuffers(1, &id);
		index = glGetUniformBlockIndex(GetShader(shaderType), name.c_str());
		glBindBuffer(GL_UNIFORM_BUFFER, id);
		glBufferData(GL_UNIFORM_BUFFER, bytes, NULL, GL_STATIC_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	RenderObject::RenderObject(const ShaderType& _type) {
		type = _type;
		unsigned int shaderId = GetShader(type);
		glGenVertexArrays(1, &id);
		glBindVertexArray(id);

		int attributesCnt;
		glGetProgramiv(shaderId, GL_ACTIVE_ATTRIBUTES, &attributesCnt);

		for (int index = 0; index < attributesCnt; index++) {
			int size;
			int length;
			GLenum attribType;
			char name[1024];
			glGetActiveAttrib(shaderId, index, 1024, &length, &size, &attribType, name);
			vbos.insert(std::make_pair(std::string(name), cv::ogl::Buffer()));
		}

		int uniformCnt;
		glGetProgramiv(shaderId, GL_ACTIVE_UNIFORMS, &uniformCnt);
		for (int index = 0; index < uniformCnt; index++) {
			int size;
			int length;
			GLenum uniformType;
			char name[1024];
			glGetActiveUniform(shaderId, index, 1024, &length, &size, &uniformType, name);
			if (name[0] == '_')				// mark the var should be set by ubo
				continue;					
			UniformVariable uniform(uniformType);
			uniform.Download(shaderId, glGetUniformLocation(shaderId, name));
			uniforms.insert(std::make_pair(std::string(name), uniform));
			if(uniformType == GL_SAMPLER_2D)
				texs.insert(std::make_pair(std::string(name), cv::ogl::Texture2D()));
		}
	}

	void RenderObject::SetBuffer(const std::string& name, cv::InputArray arr, cudaStream_t stream) {
		glBindVertexArray(id);
		if (name == "face") {
			if(stream)
				ebo.copyFrom(arr, cv::cuda::StreamAccessor::wrapStream(stream),
					cv::ogl::Buffer::Target::ELEMENT_ARRAY_BUFFER);
			else
				ebo.copyFrom(arr, cv::ogl::Buffer::Target::ELEMENT_ARRAY_BUFFER);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo.bufId());
		}
		
		else {
			auto iter = vbos.find(name);
			if (iter != vbos.end()) {
				auto& vbo = iter->second;
				if (stream)
					vbo.copyFrom(arr, cv::cuda::StreamAccessor::wrapStream(stream),
						cv::ogl::Buffer::Target::ARRAY_BUFFER);
				else
					vbo.copyFrom(arr, cv::ogl::Buffer::Target::ARRAY_BUFFER);

				glBindBuffer(GL_ARRAY_BUFFER, vbo.bufId());
				const int location = glGetAttribLocation(GetShader(type), name.c_str());
				GLenum attribType = [bufferType = vbo.type()] {
					switch (bufferType) {
					case CV_32FC1: return GL_FLOAT;
					case CV_32SC1: return GL_INT;
					case CV_8UC1: return GL_UNSIGNED_BYTE;
					default:
						std::cerr << "Unsupport Buffer Type" << std::endl;
						std::abort();
					}
				}();
				glVertexAttribPointer(location, vbo.cols(), attribType, GL_FALSE, 0, NULL);			// already bind vbo when call upload()
				glEnableVertexAttribArray(location);
			}
		}
	}

	void RenderObject::SetTexture(const std::string& name, cv::InputArray arr) {
		auto iter = texs.find(name);
		if (iter != texs.end())
			iter->second.copyFrom(arr);
	}

	void RenderObject::SetUniform(const std::string& name, void* data) {
		auto iter = uniforms.find(name);
		if (iter != uniforms.end())
			memcpy(iter->second.data.data(), data, iter->second.data.size());
	}

	void RenderObject::SetModel(const Model& model)
	{
		SetBuffer("face", model.faces);
		SetBuffer("vertex", model.vertices);
		SetBuffer("normal", model.normals);
		SetBuffer("color", model.colors);
		SetBuffer("texcoord", model.texcoords);
	}

	void RenderObject::Draw()
	{
		unsigned int shaderId = GetShader(type);
		glUseProgram(shaderId);

		// set tex
		int texIndex = 0;
		for (auto iter = texs.begin(); iter != texs.end(); iter++, texIndex++) {
			glActiveTexture(GL_TEXTURE0 + texIndex);
			iter->second.bind();
			SetUniform(iter->first, &texIndex);
		}

		// uniform var
		for (const auto& uniform : uniforms) {
			const int index = glGetUniformLocation(shaderId, uniform.first.c_str());
			if(index >= 0)
				uniform.second.Upload(shaderId, index);
		}

		// draw
		glBindVertexArray(id);
		glDrawElements(GL_TRIANGLES, ebo.rows() * ebo.cols(), GL_UNSIGNED_INT, NULL);
	}

	Viewer::Viewer()
	{
		Perspective(float(EIGEN_PI) / 4.f, 1.f, 1e-2f, 1e4f);
		LookAt(Eigen::Vector3f(0.f, 0.f, 1.f), Eigen::Vector3f::Zero(), Eigen::Vector3f(0.f, -1.f, 0.f));
	}

	void Viewer::Perspective(const float& _fov, const float& _aspect, const float& _nearest, const float& _farthest)
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

	void Viewer::LookAt(const Eigen::Vector3f& _eye, const Eigen::Vector3f& _center, const Eigen::Vector3f& _up) {
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

	void Viewer::SetCam(const Eigen::Matrix3f& K, const Eigen::Matrix3f& R, const Eigen::Vector3f& T) {
		perspective <<
			K(0, 0) * 2.f, 0.f, 0.f, 0.f,
			0.f, K(1, 1) * 2.f, 0.f, 0.f,
			0.f, 0.f, -(nearest + farthest) / (farthest - nearest), -2.f*farthest*nearest / (farthest - nearest),
			0.f, 0.f, -1.f, 0.f;

		perspective(0, 2) = 1.f - 2.f*K(0, 2);
		perspective(1, 2) = 2.f*K(1, 2) - 1.f;
		eye = -R.transpose() * T;
		center = eye + radius * R.row(2).transpose();
		LookAt(eye, center, R.row(1).transpose());
	}

	Eigen::Vector3f Viewer::ArcballMap(const Eigen::Vector2f& p) const {
		const Eigen::Vector2f pClip(2.f*p.x() - 1.f, 2.f*p.y() - 1.f);
		return pClip.squaredNorm() < 0.5f ? (Eigen::Vector3f(pClip.x(), pClip.y(), sqrtf(1.f - pClip.squaredNorm())))
			: (Eigen::Vector3f(pClip.x(), pClip.y(), 0.5f / pClip.norm())).normalized();
	};

	void Viewer::DragEye(const Eigen::Vector2f& pBegin, const Eigen::Vector2f& pEnd)
	{
		const Eigen::Vector3f posBegin = ArcballMap(pBegin);
		const Eigen::Vector3f posEnd = ArcballMap(pEnd);
		const float angle = acos(posBegin.dot(posEnd));
		Eigen::Matrix3f R;
		R << right.transpose(), up.transpose(), front.transpose();
		const Eigen::Vector3f axis = R.transpose() * posBegin.cross(posEnd).normalized();
		const Eigen::Matrix3f rot = Eigen::AngleAxisf(angle, axis).matrix();
		SetEye(rot * (eye - center) + center);
	}

	void Viewer::DragCenter(const Eigen::Vector2f& pBegin, const Eigen::Vector2f& pEnd)
	{
		const Eigen::Vector3f posBegin = ArcballMap(pBegin);
		const Eigen::Vector3f posEnd = ArcballMap(pEnd);
		const float angle = acos(posBegin.dot(posEnd));
		Eigen::Matrix3f R;
		R << right.transpose(), up.transpose(), front.transpose();
		const Eigen::Vector3f axis = R.transpose() * posBegin.cross(posEnd).normalized();
		const Eigen::Matrix3f rot = Eigen::AngleAxisf(angle, axis).matrix();
		SetCenter(rot.transpose() *(center - eye) + eye);
	}

	void Viewer::Upload()
	{
		unsigned int id = GLUtil::GetUBO(GLUtil::UBO_CAMERA);
		glBindBufferBase(GL_UNIFORM_BUFFER, GLUtil::UBO_CAMERA, id);
		glBindBuffer(GL_UNIFORM_BUFFER, id);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, 16*sizeof(float), perspective.data());
		glBufferSubData(GL_UNIFORM_BUFFER, 16 * sizeof(float), 16 * sizeof(float), transform.data());
		glBufferSubData(GL_UNIFORM_BUFFER, 32 * sizeof(float), 3 * sizeof(float), eye.data());
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}


	unsigned int CreateUBO(const int& bytes)
	{
		unsigned int id;
		glGenBuffers(1, &id);
		glBindBuffer(GL_UNIFORM_BUFFER, id);
		glBufferData(GL_UNIFORM_BUFFER, bytes, NULL, GL_STATIC_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
		return id;
	}

	unsigned int GetUBO(const UBOType& type)
	{
		switch (type)
		{
		case UBO_CAMERA:
		{static unsigned int id = CreateUBO(144); return id; }
		case SHADER_TEXTURE:
		default:
			std::cerr << "Unsupport shader type" << std::endl;
			std::abort();
			break;
		}
	}


	unsigned int CreateShader(const std::string& vs, const std::string& gs, const std::string& fs)
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

		unsigned int id = glCreateProgram();

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

		return id;
	}

	unsigned int GetShader(const ShaderType& type)
	{
		switch (type)
		{
		case SHADER_COLOR:
		{static unsigned int id = CreateShader(
			"#version 450 core\n"
			"layout(location = 0) in vec3 vertex; \n"
			"layout(location = 1) in vec3 normal; \n"
			"layout(location = 2) in vec4 color; \n"

			"out VS_OUT\n"
			"{\n"
			"	vec3 vertex; \n"
			"	vec3 normal; \n"
			"	vec4 color; \n"
			"} vs_out;\n"

			"layout(std140, binding = 0) uniform camera \n"
			"{\n"
			"mat4 _perspective;\n"
			"mat4 _transform;\n"
			"vec3 _eye;\n"
			"};\n"

			"void main()\n"
			"{\n"
			"	vs_out.vertex = vertex;\n"
			"	vs_out.normal = normal;\n"
			"	vs_out.color = color;\n"
			"	gl_Position = _perspective * _transform *vec4(vertex, 1.0);\n"
			"}\n",
			"",
			"#version 450 core\n"
			"in VS_OUT\n"
			"{\n"
			"	vec3 vertex;\n"
			"	vec3 normal;\n"
			"	vec4 color;\n"
			"}fs_in;\n"

			"out vec4 color_out;\n"
			"uniform float ambient = 0.5;\n"
			"uniform float diffuse = 0.5;\n"
			"uniform float specular = 0.2;\n"
			"uniform float shininess = 20;\n"

			"layout(std140, binding = 0) uniform camera \n"
			"{\n"
			"mat4 _perspective;\n"
			"mat4 _transform;\n"
			"vec3 _eye;\n"
			"};\n"

			"void main()\n"
			"{\n"
			"	vec3 eye_dir = normalize(_eye - fs_in.vertex);\n"
			"	vec3 light_dir = eye_dir;\n"

			"	// diffuse \n"
			"	float diff = max(dot(fs_in.normal, light_dir), 0.0) * diffuse;\n"

			"	// specular\n"
			"	vec3 reflect_dir = reflect(-light_dir, fs_in.normal);\n"
			"	float spec = pow(max(dot(eye_dir, reflect_dir), 0.0), shininess) * specular;\n"

			"	// sum\n"
			"	color_out = vec4((ambient + diff + spec) * fs_in.color.xyz, fs_in.color.w);\n"
			"}\n");
			return id;}
		case SHADER_TEXTURE:
		{static unsigned int id = CreateShader(
			"#version 450 core\n"
			"layout(location = 0) in vec3 vertex;\n"
			"layout(location = 1) in vec3 normal; \n"
			"layout(location = 3) in vec2 texcoord; \n"

			"layout(std140, binding = 0) uniform camera \n"
			"{\n"
				"mat4 _perspective;\n"
				"mat4 _transform;\n"
				"vec3 _eye;\n"
			"};\n"

			"out VS_OUT\n"
			"{\n"
			"	vec3 vertex;\n"
			"	vec3 normal;\n"
			"	vec2 texcoord;\n"
			"} vs_out;\n"

			"void main()\n"
			"{\n"
			"	vs_out.vertex = vertex;\n"
			"	vs_out.normal = normal;\n"
			"	vs_out.texcoord = texcoord;\n"
			"	gl_Position = _perspective * _transform *vec4(vertex, 1.0);\n"
			"}\n",
			"",
			"#version 450 core\n"
			"in VS_OUT\n"
			"{\n"
			"	vec3 vertex;\n"
			"	vec3 normal;\n"
			"	vec2 texcoord;\n"
			"}fs_in;\n"

			"out vec4 color_out;\n"

			"uniform sampler2D tex_uv;\n"
			"uniform float ambient = 0.5;\n"
			"uniform float diffuse = 0.5;\n"
			"uniform float specular = 0.2;\n"
			"uniform float shininess = 20;\n"

			"layout(std140, binding = 0) uniform camera  \n"
			"{\n"
				"mat4 _perspective;\n"
				"mat4 _transform;\n"
				"vec3 _eye;\n"
			"};\n"

			"void main()\n"
			"{\n"
			"	vec3 eye_dir = normalize(_eye - fs_in.vertex);\n"
			"	vec3 light_dir = eye_dir;\n"

			"	// diffuse \n"
			"	float diff = max(dot(fs_in.normal, light_dir), 0.0) * diffuse;\n"

			"	// specular\n"
			"	vec3 reflect_dir = reflect(-light_dir, fs_in.normal);\n"
			"	float spec = pow(max(dot(eye_dir, reflect_dir), 0.0), shininess) * specular;\n"

			"	// sum\n"
			"	vec4 tex_color = texture(tex_uv, fs_in.texcoord);\n"
			"	color_out = vec4((ambient + diff + spec) * tex_color.xyz, tex_color.w);\n"
			"}\n");
			return id;}
		default:
			std::cerr << "Unsupport shader type" << std::endl;
			std::abort();
			break;
		}	
	}
}
