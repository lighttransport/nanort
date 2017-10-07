#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wvariadic-macros"
#pragma clang diagnostic ignored "-Wc++11-extensions"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif

#include "embree2/rtcore.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <string>
#include <vector>
#include <cassert>
#include <sstream>

namespace nanort_embree2 {

class Scene
{
  public:
		Scene(RTCSceneFlags sflags, RTCAlgorithmFlags aflags) :
			scene_flags_(sflags),
			algorithm_flags_(aflags) {

			(void)scene_flags_;
			(void)algorithm_flags_;
		}

		~Scene() {}

		///
		/// Get scene bounding box.
		///
		void GetBounds(RTCBounds &bounds) {
			(void)bounds;
		}

  private:
		RTCSceneFlags scene_flags_;
		RTCAlgorithmFlags algorithm_flags_;
};

class Device
{
  public:
		Device(const std::string &config) :
			config_(config),
			error_func_(NULL),
			user_ptr_(NULL)
		{
		}

		~Device() {}

		void SetErrorFunction(RTCErrorFunc2 func, void *user_ptr) {
			error_func_ = func;
			user_ptr_ = user_ptr;
		}

		void AddScene(Scene *scene) {
			scenes_.push_back(scene);
		}

	private:
		std::string config_;

		std::vector<Scene *> scenes_;

		// Callbacks
		RTCErrorFunc2 error_func_;
		void *user_ptr_;

};

class Context
{
  public:
		Context() {}
		~Context() {
			for (size_t i = 0; i < devices_.size(); i++) {
				delete devices_[i];
			}
	  }

		Device *NewDevice(const char *config) {
			std::string cfg;
			if (config) {
				cfg = std::string(config);
			}

			Device *device = new Device(cfg);

			devices_.push_back(device);

			return device;
		}

		std::vector<Device *> GetDeviceList() {
			return devices_;
		}

		void SetError(const std::string &err) {
			error_ = err;
		}

 private:
	std::string error_;
	std::vector<Device*> devices_;
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif

static Context GetContext() {
	static Context s_ctx;

	return s_ctx;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

RTCORE_API RTCDevice rtcNewDevice(const char* cfg = NULL)
{
	Device *device = GetContext().NewDevice(cfg);

	return reinterpret_cast<RTCDevice>(device);
}

RTCORE_API void rtcDeleteDevice(RTCDevice device) {
	Device *dev = reinterpret_cast<Device *>(device);
	
	std::vector<Device *> devices = GetContext().GetDeviceList();

	bool found = false;
	// Simple linear search
	for (size_t i = 0; i < devices.size(); i++) {
		if (dev == devices[i]) {
			delete devices[i];
			devices.erase(devices.begin() + ptrdiff_t(i));
			found = true;
			break;
		}
	}

	if (!found) {
		std::stringstream ss;
		ss << "Invalid device : " << device << std::endl;
		GetContext().SetError(ss.str());
	}
}

RTCORE_API void rtcDeviceSetErrorFunction2(RTCDevice device, RTCErrorFunc2 func, void* userPtr)
{
	Device *ptr = reinterpret_cast<Device *>(device);
	ptr->SetErrorFunction(func, userPtr);
}

RTCORE_API RTCScene rtcDeviceNewScene (RTCDevice device, RTCSceneFlags flags, RTCAlgorithmFlags aflags)
{
	Scene *scene = new Scene(flags, aflags);

	Device *d = reinterpret_cast<Device *>(device);
	d->AddScene(scene);
	
	return reinterpret_cast<RTCScene>(scene);

}

RTCORE_API void rtcGetBounds(RTCScene scene, RTCBounds& bounds_o)
{
	Scene *s = reinterpret_cast<Scene *>(scene);
	s->GetBounds(bounds_o);
}

RTCORE_API void rtcIntersect (RTCScene scene, RTCRay& ray)
{
	Scene *s = reinterpret_cast<Scene *>(scene);

	(void)s;

	(void)ray;
}

RTCORE_API unsigned rtcNewTriangleMesh (RTCScene scene,                    //!< the scene the mesh belongs to
                                        RTCGeometryFlags flags,            //!< geometry flags
                                        size_t numTriangles,               //!< number of triangles
                                        size_t numVertices,                //!< number of vertices
                                        size_t numTimeSteps = 1            //!< number of motion blur time steps
  ) {

	(void)scene;
	(void)flags;
	(void)numTriangles;
	(void)numVertices;
	(void)numTimeSteps;

	return 0;
}

RTCORE_API void* rtcMapBuffer(RTCScene scene, unsigned geomID, RTCBufferType type) {
	(void)scene;
	(void)geomID;
	(void)type;

	return NULL;
}

RTCORE_API void rtcUnmapBuffer(RTCScene scene, unsigned geomID, RTCBufferType type) {
	(void)scene;
	(void)geomID;
	(void)type;
}


} // namespace nanort_embree2


