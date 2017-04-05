#include"Ray.h"

using namespace PTUtility;

Vec3 Ray::ReflectRay(const Ray& in, const Vec3& normal){

	return (in.m_Dir - 2.0f * (normal.dot(in.m_Dir)) * normal).normalized();

}

//in は、入ってくる方向　不透明材質なら、法線と逆方向で指定する
Vec3 Ray::RefractRay(const Ray& in, const Vec3& normal, float eta_from, float eta_to){
	return normal;
}

//in は、入ってくる方向　不透明材質なら、法線と逆方向で指定する
float Ray::FresnelReflectance(const Ray& in, const Vec3& normal){
	return 1.0f;
}

//in は、入ってくる方向　不透明材質なら、法線と逆方向で指定する
float Ray::FresnelTransmittance(const Ray& in, const Vec3& normal, float eta_from, float eta_to){
	return 1.0f;
}


Ray::Ray(const Vec3& Origin, const Vec3& Direction) : m_Org(Origin), m_Dir(Direction.normalized()), m_Index(1.0f){
}

Ray::Ray(const Vec3& Origin, const Vec3& Direction, float Index) : m_Org(Origin), m_Dir(Direction.normalized()), m_Index(Index){

}

Ray::~Ray(){

}