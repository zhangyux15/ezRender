#include <opencv2/opencv.hpp>
#include "canvas_widget.h"
#include "app.h"
#include "model.h"


int main() {
	nanogui::init();
	nanogui::ref<Application> app(new Application(Eigen::Vector2i(1200, 900), "Application"));
	app->setVisible(true);
	CanvasWidget* canvasWidget = new CanvasWidget(app, { 1200, 900 });
	app->performLayout();

	Model model("../data/model/cube.obj");
	model.Drive(Eigen::Vector3f::Constant(0.1f), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
	model.CalcNormal();

	std::shared_ptr<GLUtil::RenderObject> object = std::make_shared<GLUtil::RenderObject>(GLUtil::SHADER_TEXTURE);
	//std::shared_ptr<GLUtil::RenderObject> object = std::make_shared<GLUtil::RenderObject>(GLUtil::SHADER_COLOR);
	
	object->SetBuffer("face", model.faces);
	object->SetBuffer("vertex", model.vertices);
	object->SetBuffer("normal", model.normals);

	//model.SetColor(Eigen::Vector4f(0.8f, 0.5f, 0.6f, 1.f));
	//object->SetBuffer("color", model.colors);
	object->SetBuffer("texcoord", model.texcoords);
	object->SetTexture("tex_uv", cv::imread("../data/texture/water.jpg"));
	canvasWidget->canvas->objects.emplace_back(object);

	//vao->UploadModel(model);
	//vao->UploadTex(cv::imread("../data/texture/wood.png"), GL_BGR);

	nanogui::mainloop();
	nanogui::shutdown();
	return 0;
}

