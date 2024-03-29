#include "GLApp/GLApp.h"
#include "CLCommon/CLCommon.h"
#include "Common/File.h"
#include "Common/Macros.h"
#include "Common/Exception.h"
#include "Profiler/Profiler.h"
#include "GLCxx/gl.h"
#include "CLCommon/cl.h"

#include <Tensor/Vector.h>
#include <iostream>

//also in HydroGPU
//TODO put in Common

template<typename T> std::string toNumericString(T value);

template<> inline std::string toNumericString<double>(double value) {
	std::string s = std::to_string(value);
	if (s.find("e") == std::string::npos) {
		if (s.find(".") == std::string::npos) {
			s += ".";
		}
	}
	return s;
}

template<> inline std::string toNumericString<float>(float value) {
	return toNumericString<double>(value) + "f";
}


struct CellularApp : public GLApp::GLApp {
	typedef ::GLApp::GLApp Super;

	std::shared_ptr<CLCommon::CLCommon> clCommon;

	Tensor::vec<float,2> viewPos;
	float viewZoom;

	bool leftButtonDown;
	bool leftShiftDown;
	bool rightShiftDown;

	float aspectRatio;
		
	Tensor::vec<int, 2> size;

	GLuint texID;	
	cl::Memory texMem;

	cl::Program program;
	std::vector<cl::Buffer> updateBuffers;
	cl::Kernel updateKernel;

	int bufferIndex;

	CellularApp()
	: Super()
	, viewZoom(1)
	, leftButtonDown(false)
	, leftShiftDown(false)
	, rightShiftDown(false)
	, aspectRatio(1.f)
	, size(2048, 2048)
	, texID(0)
	, bufferIndex(0)
	{}
	
	virtual void init(GLApp::Init const & args) {
		Super::init(args);

		clCommon = std::make_shared<CLCommon::CLCommon>();

		SDL_GL_SetSwapInterval(0);

		glGenTextures(1, &texID);
		glBindTexture(GL_TEXTURE_2D, texID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, size(0), size(1), 0, GL_RGBA, GL_FLOAT, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);
		
		texMem = cl::ImageGL(clCommon->context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texID);

		{
			std::vector<std::string> sourceStrs = {
				"#define SIZE_X " + std::to_string(size(0)) + "\n"
				"#define SIZE_Y " + std::to_string(size(1)) + "\n",
				Common::File::read("update.cl")
			};
			program = cl::Program(clCommon->context, sourceStrs);
		}

		try {
			program.build({clCommon->device});
		} catch (cl::Error &err) {
			throw Common::Exception() 
				<< "failed to build program executable!\n"
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clCommon->device);
		}

		updateKernel = cl::Kernel(program, "update");

		for (int i = 0; i < 2; ++i) {
			updateBuffers.push_back(
				cl::Buffer(clCommon->context, CL_MEM_READ_WRITE, size.product() * 4 * sizeof(float))
			);
		}

#if 1	//random init
		{
			std::vector<float> initData(4 * size.product());
			for (float &i : initData) {
				i = rand() & 1;
			}
			clCommon->commands.enqueueWriteBuffer(updateBuffers[bufferIndex], CL_TRUE, 0, 4 * sizeof(float) * size.product(), &initData[0]);
		}
#endif
#if 0	//preconfigured init
		{
			std::vector<float> initData(4 * size.product());
			initData[ 0 + 4 * (size(0)/2 + size(0) * size(1)/2) ] = 1e+5f;	//full red
			clCommon->commands.enqueueWriteBuffer(updateBuffers[bufferIndex], CL_TRUE, 0, 4 * sizeof(float) * size.product(), &initData[0]);
		}
#endif
	}

	virtual void onResize() {
		Super::onResize();
		aspectRatio = (float)screenSize.x / (float)screenSize.y;
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-aspectRatio *.5, aspectRatio * .5, -.5, .5, -1., 1.);
		glMatrixMode(GL_MODELVIEW);
	}

	virtual void onUpdate() {
PROFILE_BEGIN_FRAME()
		glFinish();
	
		std::vector<cl::Memory> acquireGLMems = {texMem};
		clCommon->commands.enqueueAcquireGLObjects(&acquireGLMems);

		//run kernel
		cl::NDRange globalSize = cl::NDRange(size(0), size(1));
		cl::NDRange localSize = cl::NDRange(16, 16);
		cl::NDRange offset = cl::NDRange(0, 0);

		//update new buffer and display texture
		CLCommon::setArgs(updateKernel, updateBuffers[!bufferIndex], updateBuffers[bufferIndex], texMem);
		clCommon->commands.enqueueNDRangeKernel(updateKernel, offset, globalSize, localSize);
		bufferIndex = !bufferIndex;
	
		clCommon->commands.enqueueReleaseGLObjects(&acquireGLMems);
		clCommon->commands.finish();

		//clear screen buffer
		Super::onUpdate();
	
		//transforms
		glLoadIdentity();
		glTranslatef(-viewPos(0), -viewPos(1), 0);
		glScalef(viewZoom, viewZoom, viewZoom);
	
		//display
		glBindTexture(GL_TEXTURE_2D, texID);
		glEnable(GL_TEXTURE_2D);
		glBegin(GL_QUADS);
		glTexCoord2f(0,0); glVertex2f(-.5f, -.5f);
		glTexCoord2f(1,0); glVertex2f(.5f, -.5f);
		glTexCoord2f(1,1); glVertex2f(.5f, .5f);
		glTexCoord2f(0,1); glVertex2f(-.5f, .5f);
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);	
PROFILE_END_FRAME()
	}

	virtual void sdlEvent(SDL_Event& event) {
		bool shiftDown = leftShiftDown || rightShiftDown;
	
		switch (event.type) {
		case SDL_MOUSEMOTION:
			{
				int dx = event.motion.xrel;
				int dy = event.motion.yrel;
				if (leftButtonDown) {
					if (shiftDown) {
						if (dy) {
							float scale = exp((float)dy * -.03f);
							viewPos *= scale;
							viewZoom *= scale;
						} 
					} else {
						if (dx || dy) {
							viewPos += Tensor::vec<float,2>(-(float)dx * aspectRatio / (float)screenSize(0), (float)dy / (float)screenSize(1));
						}
					}
				}
			}
			break;
		case SDL_MOUSEBUTTONDOWN:
			if (event.button.button == SDL_BUTTON_LEFT) {
				leftButtonDown = true;
			}
			break;
		case SDL_MOUSEBUTTONUP:
			if (event.button.button == SDL_BUTTON_LEFT) {
				leftButtonDown = false;
			}
			break;
		case SDL_KEYDOWN:
			if (event.key.keysym.sym == SDLK_LSHIFT) {
				leftShiftDown = true;
			} else if (event.key.keysym.sym == SDLK_RSHIFT) {
				rightShiftDown = true;
			}
			break;
		case SDL_KEYUP:
			if (event.key.keysym.sym == SDLK_LSHIFT) {
				leftShiftDown = false;
			} else if (event.key.keysym.sym == SDLK_RSHIFT) {
				rightShiftDown = false;
			}
		}
	}
};

GLAPP_MAIN(CellularApp)
