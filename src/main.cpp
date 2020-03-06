#include "canvas_widget.h"
#include <opencv2/opencv.hpp>


struct ExampleApplication: public nanogui::Screen
{
	ExampleApplication() : nanogui::Screen(Eigen::Vector2i(1200, 900), "Application") {
		using namespace nanogui;
		nanogui::Window *window = new Window(this, "Canvas Demo");
		window->setLayout(new GridLayout());
		performLayout();
	}

	virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) {
		if (Screen::keyboardEvent(key, scancode, action, modifiers))
			return true;
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
			setVisible(false);
			return true;
		}
		return false;
	}

	virtual void draw(NVGcontext *ctx) {
		/* Draw the user interface */
		Screen::draw(ctx);
	}
};


int main() {
	nanogui::init();
	nanogui::ref<ExampleApplication> app(new ExampleApplication());
	app->setVisible(true);
	CanvasWidget* canvasWidget = new CanvasWidget(app, { 900, 900 });
	app->performLayout();

	GLUtil::Model model("../data/model/cube.obj");
	model.Drive(Eigen::Vector3f::Constant(0.1f), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
	std::shared_ptr<GLUtil::VAO> vao = std::make_shared<GLUtil::VAO>();
	vao->UploadModel(model);
	vao->UploadTex(cv::imread("../data/texture/wood.png"));
	canvasWidget->canvas->vaos.emplace_back(vao);

	nanogui::mainloop();
	nanogui::shutdown();
	return 0;
}

