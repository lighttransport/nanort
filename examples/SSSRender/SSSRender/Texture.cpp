#include"Texture.h"

Texture2D::Texture2D() : m_Comp(0), m_X(0), m_Y(0){

}
Texture2D::~Texture2D(){

}

bool Texture2D::LoadTextureFromFile(const char* FileName){

	if (FileName == ""){
		return false;
	}
	if (FileName == nullptr){
		return false;
	}
	unsigned char* data = nullptr;
	data = stbi_load(FileName, &m_X, &m_Y, &m_Comp, 0);

	if (data == nullptr){
		return false;
	}

	int compmin = std::min(3, m_Comp);

	//Load Image to m_Image;
	for (int i = 0; i < m_X; ++i){
		std::vector<PTUtility::Vec3> cA;
		for (int j = 0; j < m_Y; ++j){
			PTUtility::Vec3 cl(0, 0, 0);
			int IDX = j*m_X + i;
			for (int c = 0; c < compmin; ++c){
				cl[c] = data[IDX * compmin + c];
			}
			cA.push_back(cl);
		}
		m_Image.push_back(cA);
	}
	stbi_image_free(data);
	return true;
}

PTUtility::Vec3 Texture2D::GetColor(float u, float v)const{
	return m_LinearSampler.Sample(m_Image, u, v) / (255.0f);
}