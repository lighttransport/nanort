#pragma once
#include <chrono>
#include<string>
#include<vector>
#include<ostream>

/*
std::vector<ScopedTimer::ID_TIME> ScopedTimer::m_Table;
*/

//Timer Utility
class ScopedTimer{

public:
	struct ID_TIME{
		std::string _ID;
		double _PassedTime;
		ID_TIME(const std::string& ID) : _ID(ID), _PassedTime(0.0){}
	};

private:
	static std::vector<ID_TIME> m_Table;
	ID_TIME m_result;

	std::chrono::system_clock::time_point m_Start;

public:
	ScopedTimer(const std::string& ID) : m_result(ID), m_Start(std::chrono::system_clock::now()){
	}
	virtual ~ScopedTimer(){
		std::chrono::system_clock::time_point End = std::chrono::system_clock::now();
		m_result._PassedTime = std::chrono::duration_cast<std::chrono::milliseconds>(End - m_Start).count();
		if (m_Table.size() < 100){
			m_Table.push_back(m_result);
		}
	}

	static std::vector<ID_TIME>& GetTimeTable(){
		return m_Table;
	}
	static void PrintTimeTable(std::ostream& Out){
		for (const auto& t : m_Table){
			Out << "ID:" << t._ID << " --- " << "PassedTime(msec):" << t._PassedTime << std::endl;
		}
	}
};

