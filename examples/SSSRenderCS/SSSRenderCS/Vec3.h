#pragma once
#include<cmath>

namespace PTUtility{

    //x,y,z vector class 
	class Vec3{

	private:

		float a[3];

	public:

		static Vec3 Ones(){
			return Vec3(1, 1, 1);
		}
		static Vec3 Zero(){
			return Vec3(0, 0, 0);
		}
		static Vec3 MultVec(const Vec3& left, const Vec3& right){
			return Vec3(left.x() * right.x(), left.y() * right.y(), left.z() * right.z());
		}
		static Vec3 DivVec(const Vec3& left, const Vec3& right){
			return Vec3(left.x() / right.x(), left.y() / right.y(), left.z() / right.z());
		}
		Vec3 MultVec(const Vec3& ref){
			return *this = MultVec(*this, ref);
		}
		Vec3 DivVec(const Vec3& ref){
			return *this = DivVec(*this, ref);
		}

		float dot(const Vec3& ref)const{
			return a[0] * ref.x() + a[1] * ref.y() + a[2] * ref.z();
		}
		Vec3 cross(const Vec3& ref)const{
			return Vec3(
				a[1] * ref.z() - a[2] * ref.y(),
				a[2] * ref.x() - a[0] * ref.z(),
				a[0] * ref.y() - a[1] * ref.x());
		}


		float norm()const{
			return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
		}
		float norm2()const{
			return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
		}
	
		void normalize(){
			float d = norm();
			a[0] /= d;
			a[1] /= d;
			a[2] /= d;
		}
		Vec3 normalized()const{
			float d = norm();
			return Vec3(a[0] / d, a[1] / d, a[2] / d);
		}


		inline const Vec3 operator+(const Vec3& ref)const{
			return Vec3(a[0] + ref.x(), a[1] + ref.y(), a[2] + ref.z());
		}
		inline const Vec3 operator-(const Vec3& ref)const{
			return Vec3(a[0] - ref.x(), a[1] - ref.y(), a[2] - ref.z());
		}
		inline const Vec3 operator-()const{
			return Vec3(-a[0], -a[1], -a[2]);
		}
		inline const Vec3 operator+()const{
			return *this;
		}



		inline float x()const{
			return a[0];
		}
		inline float y()const{
			return a[1];
		}
		inline float z()const{
			return a[2];
		}
		inline float& x(){
			return a[0];
		}
		inline float& y(){
			return a[1];
		}
		inline float& z(){
			return a[2];
		}

		inline float operator[](size_t id)const{
			return a[id];
		}
		inline float& operator[](size_t id){
			return a[id];
		}

		inline float operator()(size_t id)const{
			return a[id];
		}
		inline float& operator()(size_t id){
			return a[id];
		}

		inline Vec3& operator+=(const Vec3& ref){
			a[0] += ref.x();
			a[1] += ref.y();
			a[2] += ref.z();
			return *this;
		}
		inline Vec3& operator-=(const Vec3& ref){
			a[0] -= ref.x();
			a[1] -= ref.y();
			a[2] -= ref.z();
			return *this;
		}


		Vec3(const Vec3& ref){
			a[0] = ref.x();
			a[1] = ref.y();
			a[2] = ref.z();
		}
		Vec3& operator=(const Vec3& ref){
			a[0] = ref.x();
			a[1] = ref.y();
			a[2] = ref.z();
			return *this;
		}

		Vec3(){
			a[0] = a[1] = a[2] = 0.0f;
		}
		Vec3(float x, float y, float z){
			a[0] = x;
			a[1] = y;
			a[2] = z;
		}


		~Vec3(){

		}

	};


	inline const Vec3 operator*(float left, const Vec3& right){
		return Vec3(left*right.x(), left*right.y(), left*right.z());
	}
	inline const Vec3 operator*(const Vec3& left, float right){
		return Vec3(right*left.x(), right*left.y(), right*left.z());
	}
	inline const Vec3 operator*=(Vec3& left, float right){
		left.x() *= right;
		left.y() *= right;
		left.z() *= right;
		return left;
	}
	inline const Vec3 operator/=(Vec3& left, float right){
		left.x() /= right;
		left.y() /= right;
		left.z() /= right;
		return left;
	}

	inline const Vec3 operator/(const Vec3& left, float right){
		return Vec3(left.x() / right, left.y() / right, left.z() / right);
	}


}