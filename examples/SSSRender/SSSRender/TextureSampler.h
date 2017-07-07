#pragma once
#include<vector>
#include<algorithm>

namespace Sampler{
	template<typename T>
	class iSampler{
	private:
		//iSampler& operator=(const iSampler&);
		//iSampler(const iSampler&);
	public:
		iSampler(){}
		virtual T Sample(const std::vector<std::vector<T>>& texture, float u, float v)const = 0;
	};

	template<typename T>
	class PointSampler : public iSampler<T>{
	private:
	public:
		PointSampler(){}

		virtual T Sample(const std::vector<std::vector<T>>& texture, float u, float v)const;
		virtual ~PointSampler();
	};

	template<typename T>
	class LinearSampler : public iSampler<T>{
	private:
	public:
		LinearSampler(){}
		virtual T Sample(const std::vector<std::vector<T>>& texture, float u, float v)const;
		virtual ~LinearSampler();
	};

}

template<typename T>
Sampler::PointSampler<T>::~PointSampler(){

}

template<typename T>
T Sampler::PointSampler<T>::Sample(const std::vector<std::vector<T>>& texture, float u, float v)const{
	int Width = texture[0].size();
	int Height = texture.size();

	int x = std::min<int>(std::max<int>(0, Width * u + 0.5), Width - 1);
	int y = std::min<int>(std::max<int>(0, Height *v + 0.5), Height - 1);

	return texture[y][x];
}



template<typename T>
Sampler::LinearSampler<T>::~LinearSampler(){
}

template<typename T>
T Sampler::LinearSampler<T>::Sample(const std::vector<std::vector<T>>& texture, float u, float v)const{
	int Width = texture[0].size();
	int Height = texture.size();

	int x1 = std::min<int>(std::max<int>(0, Width * v - 0.00001), Width - 1);
	int y1 = std::min<int>(std::max<int>(0, Height *u - 0.00001), Height - 1);

	const T& t_00 = texture[y1][x1];
	const T& t_01 = texture[std::min<int>(Height - 1, y1 + 1)][x1];
	const T& t_10 = texture[y1][std::min<int>(Width - 1, x1 + 1)];
	const T& t_11 = texture[std::min<int>(Height - 1, y1 + 1)][std::min<int>(Width - 1, x1 + 1)];

	float w = v * Width - x1;
	float h = u * Height - y1;

	T up = t_00 * (1.0 - w) + t_10 * (w);
	T down = t_01 * (1.0 - w) + t_11 * (w);

	return up * (1.0 - h) + down *(h);
}