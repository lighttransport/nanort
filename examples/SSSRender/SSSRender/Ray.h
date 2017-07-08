#pragma once
#include"Vec3.h"

namespace PTUtility{

	struct Ray{

	public:

		Vec3 m_Org;
		Vec3 m_Dir;
		float m_Index;

		Ray operator-()const{
			return Ray(this->m_Org - 0.0001 * this->m_Dir, -this->m_Dir, this->m_Index);
		}
		
		//in は、入ってくる方向　不透明材質なら、法線と逆方向で指定する
		static Vec3 ReflectRay(const Ray& in, const Vec3& normal);

		//in は、入ってくる方向　不透明材質なら、法線と逆方向で指定する
		static Vec3 RefractRay(const Ray& in, const Vec3& normal, float eta_from, float eta_to);

		//in は、入ってくる方向　不透明材質なら、法線と逆方向で指定する
		static float FresnelReflectance(const Ray& in, const Vec3& normal);

		//in は、入ってくる方向　不透明材質なら、法線と逆方向で指定する
		static float FresnelTransmittance(const Ray& in, const Vec3& normal, float eta_from, float eta_to);

		Ray(const Vec3& Origin, const Vec3& Direction);
		Ray(const Vec3& Origin, const Vec3& Direction, float Index);
		virtual ~Ray();

	};

	

}
