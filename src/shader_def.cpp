#include "gl_util.h"


namespace GLUtil
{
	std::shared_ptr<Shader> Shader::Get(const ShaderType& type)
	{
		switch (type)
		{
		case SHADER_COLOR:
		{static std::shared_ptr<Shader> shader = [] {
			const std::string vs =
				"#version 410 core\n"
				"layout(location = 0) in vec3 vertex_in; \n"
				"layout(location = 1) in vec3 normal_in; \n"
				"layout(location = 2) in vec4 color_in; \n"

				"uniform mat4 u_perspective;\n"
				"uniform mat4 u_transform;\n"

				"out VS_OUT\n"
				"{\n"
				"	vec3 vertex; \n"
				"	vec3 normal; \n"
				"	vec4 color; \n"
				"} vs_out;\n"

				"void main()\n"
				"{\n"
				"	vs_out.vertex = vertex_in;\n"
				"	vs_out.normal = normal_in;\n"
				"	vs_out.color = color_in;\n"
				"	gl_Position = u_perspective * u_transform *vec4(vertex_in, 1.0);\n"
				"}\n";

			const std::string fs =
				"#version 410 core\n"
				"in VS_OUT\n"
				"{\n"
				"	vec3 vertex;\n"
				"	vec3 normal;\n"
				"	vec4 color;\n"
				"}fs_in;\n"

				"out vec4 color_out;\n"
				"uniform vec3 u_eye;\n"
				"uniform float u_ambient;\n"
				"uniform float u_diffuse;\n"
				"uniform float u_specular;\n"
				"uniform float u_shininess;\n"

				"void main()\n"
				"{\n"
				"	vec3 eye_dist = u_eye - fs_in.vertex;\n"
				"	vec3 eye_dir = normalize(eye_dist);\n"
				"	vec3 light_dir = eye_dir;\n"

				"	// ambient\n"
				"	float ambient = u_ambient;\n"

				"	// diffuse \n"
				"	float diff = max(dot(fs_in.normal, light_dir), 0.0);\n"
				"	float diffuse = diff * u_diffuse;\n"

				"	// specular\n"
				"	vec3 reflect_dir = reflect(-light_dir, fs_in.normal);\n"
				"	vec3 halfway_dir = normalize(light_dir + eye_dir);\n"
				"	float spec = pow(max(dot(fs_in.normal, halfway_dir), 0.0), u_shininess);\n"
				"	float specular = spec * u_specular;\n"

				"	// sum\n"
				"	color_out = vec4((ambient + diffuse + specular) * fs_in.color.xyz, fs_in.color.w);\n"
				"}\n";
			return std::make_shared<Shader>(vs, "", fs);
		}();
		return shader;
		break;
		}
		case SHADER_TEXTURE:
		{static std::shared_ptr<Shader> shader = [] {
			const std::string vs =
				"#version 410 core\n"
				"layout(location = 0) in vec3 vertex_in;\n"
				"layout(location = 1) in vec3 normal_in; \n"
				"layout(location = 3) in vec2 texcoord_in; \n"

				"uniform mat4 u_perspective;\n"
				"uniform mat4 u_transform;\n"

				"out VS_OUT\n"
				"{\n"
				"	vec3 vertex;\n"
				"	vec3 normal;\n"
				"	vec2 texcoord;\n"
				"} vs_out;\n"

				"void main()\n"
				"{\n"
				"	vs_out.vertex = vertex_in;\n"
				"	vs_out.normal = normal_in;\n"
				"	vs_out.texcoord = texcoord_in;\n"
				"	gl_Position = u_perspective * u_transform *vec4(vertex_in, 1.0);\n"
				"}\n";

			const std::string fs =
				"#version 410 core\n"
				"in VS_OUT\n"
				"{\n"
				"	vec3 vertex;\n"
				"	vec3 normal;\n"
				"	vec2 texcoord;\n"
				"}fs_in;\n"

				"out vec4 color_out;\n"
				"uniform sampler2D u_tex_uv;\n"
				"uniform vec3 u_eye;\n"
				"uniform float u_ambient;\n"
				"uniform float u_diffuse;\n"
				"uniform float u_specular;\n"
				"uniform float u_shininess;\n"

				"void main()\n"
				"{\n"
				"	vec3 eye_dist = u_eye - fs_in.vertex;\n"
				"	vec3 eye_dir = normalize(eye_dist);\n"
				"	vec3 light_dir = eye_dir;\n"

				"	// ambient\n"
				"	float ambient = u_ambient;\n"

				"	// diffuse \n"
				"	float diff = max(dot(fs_in.normal, light_dir), 0.0);\n"
				"	float diffuse = diff * u_diffuse;\n"

				"	// specular\n"
				"	vec3 reflect_dir = reflect(-light_dir, fs_in.normal);\n"
				"	vec3 halfway_dir = normalize(light_dir + eye_dir);\n"
				"	float spec = pow(max(dot(fs_in.normal, halfway_dir), 0.0), u_shininess);\n"
				"	float specular = spec * u_specular;\n"

				"	// sum\n"
				"	vec4 tex_color = texture(u_tex_uv, fs_in.texcoord);\n"
				"	color_out = vec4((ambient + diffuse + specular) * tex_color.xyz, tex_color.w);\n"
				"}\n";
			return std::make_shared<Shader>(vs, "", fs);
		}();
		return shader;
		break; }
		default:
			std::cerr << "Unsupport shader type" << std::endl;
			std::abort();
			break;
		}	
	}
}
