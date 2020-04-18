#pragma once
#include <nanogui/nanogui.h>
#include <queue>
#include <mutex>

struct Application : public nanogui::Screen
{
	Application(const Eigen::Vector2i &size, const std::string &caption, bool resizable = false, bool fullscreen = false)
		: nanogui::Screen(size, caption, resizable, fullscreen) { }
	~Application() {
		while (!tasks.empty())
			tasks.pop();
	}

	void PushTask(const std::function<void()>& task)
	{
		std::unique_lock<std::mutex> locker(mu);
		tasks.push(task);
		locker.unlock();
	}

	virtual void drawAll() override { 
		std::unique_lock<std::mutex> locker(mu);
		while (!tasks.empty()) {
			tasks.front()();
			tasks.pop();
		}
		locker.unlock();
		nanogui::Screen::drawAll(); 
	}

	virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) override{
		if (Screen::keyboardEvent(key, scancode, action, modifiers))
			return true;
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
			setVisible(false);
			return true;
		}
		return false;
	}

	std::queue<std::function<void()>> tasks;
	std::mutex mu;
};

