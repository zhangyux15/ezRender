#include "canvas_widget.h"


CanvasWidget::Canvas::Canvas(nanogui::Widget* parent, const Eigen::Vector2i& size) :nanogui::GLCanvas(parent) {
	setSize(size);
	setBackgroundColor(Eigen::Vector4f::Zero());
	viewer.SetAspect(float(size.x()) / float(size.y()));
}

void CanvasWidget::Canvas::drawGL() {
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	viewer.Upload();
	for (auto object : objects)
		object->Draw();
}

cv::Mat CanvasWidget::Canvas::GetImage()
{
	this->screen()->drawAll();
	glfwSwapBuffers(this->screen()->glfwWindow());
	cv::Mat image(height(), width(), CV_8UC3);
	glReadPixels(0, 0, width(), height(), GL_BGR, GL_UNSIGNED_BYTE, image.data);
	cv::flip(image, image, 0);
	return image;
}

bool CanvasWidget::Canvas::mouseDragEvent(const Eigen::Vector2i &p, const Eigen::Vector2i &rel, int button, int modifiers)
{
	const Eigen::Vector2f pEnd = p.cast<float>().array() / size().cast<float>().array();
	const  Eigen::Vector2f pBegin = (p - rel).cast<float>().array() / size().cast<float>().array();

	switch (button)
	{
	case 1:
		viewer.DragEye(pBegin, pEnd);
		break;
	case 2:
		viewer.DragCenter(pBegin, pEnd);
		break;
	default:
		break;
	}
	return true;
}

bool CanvasWidget::Canvas::scrollEvent(const Eigen::Vector2i &p, const Eigen::Vector2f &rel)
{
	viewer.SetRadius((1.f - 0.1f * rel.y())*viewer.radius);
	return true;
}

CanvasWidget::CanvasHelper::CanvasHelper(Canvas* _canvas)
	:Widget(_canvas->parent())
{
	canvas = _canvas;
	this->setLayout(new nanogui::GridLayout());
	new nanogui::Label(this, "Width:", "sans-bold");
	widthBox = new nanogui::IntBox<unsigned int>(this);
	widthBox->setFixedSize(Eigen::Vector2i(80, 20));
	widthBox->setEditable(true);
	widthBox->setSpinnable(true);
	widthBox->setValue(canvas->width());
	widthBox->setCallback([this](const unsigned int& width) {
		canvas->setSize({ width, canvas->height() });
		canvas->viewer.SetAspect(float(width) / float(canvas->height()));
		this->screen()->performLayout();
	});

	new nanogui::Label(this, "Height:", "sans-bold");
	heightBox = new nanogui::IntBox<unsigned int>(this);
	heightBox->setFixedSize(Eigen::Vector2i(80, 20));
	heightBox->setEditable(true);
	heightBox->setSpinnable(true);
	heightBox->setValue(canvas->height());
	heightBox->setCallback([this](const unsigned int& height) {
		canvas->setSize({ canvas->width(), height });
		canvas->viewer.SetAspect(float(canvas->width()) / float(height));
		this->screen()->performLayout();
	});

	new nanogui::Label(this, "Background:", "sans-bold");
	nanogui::ColorWheel* colorWheel = new nanogui::ColorWheel(this, canvas->backgroundColor());
	colorWheel->setCallback([canvas = canvas](const nanogui::Color& color) { canvas->setBackgroundColor(color); });

	new nanogui::Label(this, "Fovy:", "sans-bold");
	fovBox = new nanogui::FloatBox<float>(this);
	fovBox->setFixedSize(Eigen::Vector2i(80, 20));
	fovBox->setEditable(true);
	fovBox->setSpinnable(true);
	fovBox->setMinMaxValues(0.f, float(EIGEN_PI));
	fovBox->setCallback([&viewer = canvas->viewer](const float& fov) { viewer.SetFov(fov); });

	new nanogui::Label(this, "Aspect:", "sans-bold");
	aspectBox = new nanogui::FloatBox<float>(this);
	aspectBox->setFixedSize(Eigen::Vector2i(80, 20));
	aspectBox->setEditable(true);
	aspectBox->setSpinnable(true);
	aspectBox->setMinValue(0.f);
	aspectBox->setCallback([&viewer = canvas->viewer](const float& aspect) { viewer.SetAspect(aspect); });

	new nanogui::Label(this, "Nearest:", "sans-bold");
	nearestBox = new nanogui::FloatBox<float>(this);
	nearestBox->setFixedSize(Eigen::Vector2i(80, 20));
	nearestBox->setEditable(true);
	nearestBox->setCallback([&viewer = canvas->viewer](const float& nearest) {viewer.SetNearest(nearest); });

	new nanogui::Label(this, "Farthest:", "sans-bold");
	farthestBox = new nanogui::FloatBox<float>(this);
	farthestBox->setFixedSize(Eigen::Vector2i(80, 20));
	farthestBox->setEditable(true);
	farthestBox->setCallback([&viewer = canvas->viewer](const float& farthest) { viewer.SetFarthest(farthest); });

	new nanogui::Label(this, "Radius:", "sans-bold");
	radiusBox = new nanogui::FloatBox<float>(this);
	radiusBox->setFixedSize(Eigen::Vector2i(80, 20));
	radiusBox->setEditable(true);
	radiusBox->setSpinnable(true);
	radiusBox->setMinValue(0.1f);
	radiusBox->setCallback([&viewer = canvas->viewer](const float& radius) { viewer.SetRadius(radius); });

	new nanogui::Label(this, "Eye:", "sans-bold");
	eyexBox = new nanogui::FloatBox<float>(this);
	eyexBox->setFixedSize(Eigen::Vector2i(80, 20));
	eyexBox->setEnabled(false);
	new nanogui::Widget(this);
	eyeyBox = new nanogui::FloatBox<float>(this);
	eyeyBox->setFixedSize(Eigen::Vector2i(80, 20));
	eyeyBox->setEnabled(false);
	new nanogui::Widget(this);
	eyezBox = new nanogui::FloatBox<float>(this);
	eyezBox->setFixedSize(Eigen::Vector2i(80, 20));
	eyezBox->setEnabled(false);

	new nanogui::Label(this, "Center:", "sans-bold");
	centerxBox = new nanogui::FloatBox<float>(this);
	centerxBox->setFixedSize(Eigen::Vector2i(80, 20));
	centerxBox->setEnabled(false);
	new nanogui::Widget(this);
	centeryBox = new nanogui::FloatBox<float>(this);
	centeryBox->setFixedSize(Eigen::Vector2i(80, 20));
	centeryBox->setEnabled(false);
	new nanogui::Widget(this);
	centerzBox = new nanogui::FloatBox<float>(this);
	centerzBox->setFixedSize(Eigen::Vector2i(80, 20));
	centerzBox->setEnabled(false);

	new nanogui::Label(this, "Up:", "sans-bold");
	upxBox = new nanogui::FloatBox<float>(this);
	upxBox->setFixedSize(Eigen::Vector2i(80, 20));
	upxBox->setEnabled(false);
	new nanogui::Widget(this);
	upyBox = new nanogui::FloatBox<float>(this);
	upyBox->setFixedSize(Eigen::Vector2i(80, 20));
	upyBox->setEnabled(false);
	new nanogui::Widget(this);
	upzBox = new nanogui::FloatBox<float>(this);
	upzBox->setEnabled(false);
	upzBox->setFixedSize(Eigen::Vector2i(80, 20));

	new nanogui::Label(this, "LookAt:", "sans-bold");
	lookAtCheckbox = new nanogui::CheckBox(this, "Locked");
	lookAtCheckbox->setFixedSize(Eigen::Vector2i(80, 20));
	lookAtCheckbox->setChecked(true);
	lookAtCheckbox->setCallback([this](const bool& checked) {
		lookAtCheckbox->setCaption(checked ? "Locked" : "Set");
		if (checked) {
			for (auto box : { eyexBox, eyeyBox, eyezBox,centerxBox, centeryBox, centerzBox,upxBox, upyBox, upzBox }) {
				box->setSpinnable(false);
				box->setEnabled(false);
				box->setEditable(false);
				canvas->viewer.LookAt(
					Eigen::Vector3f(eyexBox->value(), eyeyBox->value(), eyezBox->value()),
					Eigen::Vector3f(centerxBox->value(), centeryBox->value(), centerzBox->value()),
					Eigen::Vector3f(upxBox->value(), upyBox->value(), upzBox->value()).normalized());
			}
			freeviewBox->setEditable(true);
			freeviewBox->setEnabled(true);
		}
		else {
			for (auto box : { eyexBox, eyeyBox, eyezBox,centerxBox, centeryBox, centerzBox,upxBox, upyBox, upzBox }) {
				box->setSpinnable(true);
				box->setEnabled(true);
				box->setEditable(true);
			}
			freeviewBox->setValue(0.f);
			freeviewBox->setEditable(false);
			freeviewBox->setEnabled(false);
		}
	});

	new nanogui::Label(this, "Freeview:", "sans-bold");
	freeviewBox = new nanogui::FloatBox<float>(this);
	freeviewBox->setFixedSize(Eigen::Vector2i(80, 20));
	freeviewBox->setValue(0.f);
	freeviewBox->setSpinnable(true);
	freeviewBox->setValueIncrement(0.01f);
	freeviewBox->setUnits("/s");
	freeviewBox->setCallback([this](const float& rate) {stamp = std::chrono::steady_clock::now(); });
}


void CanvasWidget::CanvasHelper::draw(NVGcontext *ctx)
{
	if (canvas->visible()) {
		fovBox->setValue(canvas->viewer.fov);
		aspectBox->setValue(canvas->viewer.aspect);
		nearestBox->setValue(canvas->viewer.nearest);
		farthestBox->setValue(canvas->viewer.farthest);
		radiusBox->setValue(canvas->viewer.radius);

		if (lookAtCheckbox->checked()) {
			eyexBox->setValue(canvas->viewer.eye.x());
			eyeyBox->setValue(canvas->viewer.eye.y());
			eyezBox->setValue(canvas->viewer.eye.z());
			centerxBox->setValue(canvas->viewer.center.x());
			centeryBox->setValue(canvas->viewer.center.y());
			centerzBox->setValue(canvas->viewer.center.z());
			upxBox->setValue(canvas->viewer.up.x());
			upyBox->setValue(canvas->viewer.up.y());
			upzBox->setValue(canvas->viewer.up.z());
		}
		if (std::abs(freeviewBox->value()) > FLT_EPSILON) {
			const float elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(
				std::chrono::steady_clock::now() - stamp).count();
			if (elapsed > 0.1f) {
				GLUtil::Viewer& viewer = canvas->viewer;
				const float theta = 2.f * float(EIGEN_PI) * elapsed * freeviewBox->value();
				viewer.SetEye(Eigen::AngleAxisf(theta, viewer.up).matrix() * (viewer.eye - viewer.center) + viewer.center);
				stamp = std::chrono::steady_clock::now();
			}
		}
	}
	nanogui::Widget::draw(ctx);
}


CanvasWidget::CanvasWidget(nanogui::Widget* parent, const Eigen::Vector2i& size) : Widget(parent)
{
	this->setLayout(new nanogui::GridLayout(nanogui::Orientation::Horizontal, 2, nanogui::Alignment::Minimum));
	canvas = new Canvas(this, size);
	helper = new CanvasHelper(canvas);
}


bool CanvasWidget::mouseButtonEvent(const Eigen::Vector2i &p, int button, bool down, int modifiers)
{
	if (button == 2 && down) {
		helper->setVisible(!helper->visible());
		this->screen()->performLayout();
	}
	return nanogui::Widget::mouseButtonEvent(p,button, down, modifiers);
}

