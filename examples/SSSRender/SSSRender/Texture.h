#pragma once
#include"TextureSampler.h"
#include"stb_image.hpp"
#include"Vec3.h"
#include<vector>

class iTexture{
private:
public:
	virtual bool LoadTextureFromFile(const char* FileName) = 0;

	iTexture(){
	}
	virtual ~iTexture(){

	}
};

class Texture2D : public iTexture{
private:
	int m_X;
	int m_Y;
	int m_Comp;
	Sampler::LinearSampler<PTUtility::Vec3> m_LinearSampler;
	std::vector<std::vector<PTUtility::Vec3>> m_Image;

public:
	virtual bool LoadTextureFromFile(const char* FileName);
	PTUtility::Vec3 GetColor(float u, float v)const;

	Texture2D();
	virtual ~Texture2D();
};