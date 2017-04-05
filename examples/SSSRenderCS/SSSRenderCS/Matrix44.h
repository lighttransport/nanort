#pragma once
#include"Vec3.h"


namespace PTUtility{

	class Matrix44{

	private:

		float mat[4][4];


	public:

		Matrix44(){

		}
		Matrix44(
			const Vec3& xyz1, float w1,
			const Vec3& xyz2, float w2,
			const Vec3& xyz3, float w3,
			const Vec3& xyz4, float w4,
			){

		}
		Matrix44(
			float a11, float a12, float a13, float a14,
			float a21, float a22, float a23, float a24,
			float a31, float a32, float a33, float a34,
			float a41, float a42, float a43, float a44
			){

		}


		~Matrix44(){


		}


	};

}
