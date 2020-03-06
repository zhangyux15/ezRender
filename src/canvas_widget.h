#include <nanogui/nanogui.h>
#include <chrono>
#include "gl_util.h"


struct CanvasWidget : public nanogui::Widget
{
	struct Canvas : public nanogui::GLCanvas
	{
		GLUtil::Viewer viewer;
		std::vector<std::shared_ptr<GLUtil::VAO>> vaos;
		GLUtil::FPSGauge fpsGauge;

		Canvas(nanogui::Widget* parent, const Eigen::Vector2i& size);
		virtual void drawGL() override;
		virtual bool mouseDragEvent(const Eigen::Vector2i &p, const Eigen::Vector2i &rel, int button, int modifiers) override;
		virtual bool scrollEvent(const Eigen::Vector2i &p, const Eigen::Vector2f &rel) override;
	}*canvas;

	struct CanvasHelper : public nanogui::Widget
	{
		CanvasHelper(Canvas* _canvas);
		virtual void draw(NVGcontext *ctx) override;
		Canvas* canvas;
		nanogui::IntBox<unsigned int> *widthBox, *heightBox;
		nanogui::FloatBox<float> *fpsBox, *fovBox, *aspectBox, *nearestBox, *farthestBox, *radiusBox,
			*eyexBox, *eyeyBox, *eyezBox, *centerxBox, *centeryBox, *centerzBox, *upxBox, *upyBox, *upzBox, *freeviewBox;
		nanogui::CheckBox *lookAtCheckbox;
		std::chrono::time_point<std::chrono::steady_clock> stamp;
	}*helper;

	CanvasWidget(nanogui::Widget* parent, const Eigen::Vector2i& size);
	virtual bool mouseButtonEvent(const Eigen::Vector2i &p, int button, bool down, int modifiers) override;
	cv::Mat GetImage();
};
