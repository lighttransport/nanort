// https://gist.github.com/galloscript/8a5d179e432e062550972afcd1ecf112

//
//  imgui_color_gradient.h
//  imgui extension
//
//  Created by David Gallardo on 11/06/16.

/*
 
 Usage:
 
 ::GRADIENT DATA::
 ImGradient gradient;
 
 ::BUTTON::
 if(ImGui::GradientButton(&gradient))
 {
    //set show editor flag to true/false
 }
 
 ::EDITOR::
 static ImGradientMark* draggingMark = nullptr;
 static ImGradientMark* selectedMark = nullptr;
 
 bool updated = ImGui::GradientEditor(&gradient, draggingMark, selectedMark);
 
 ::GET A COLOR::
 float color[3];
 gradient.getColorAt(0.3f, color); //position from 0 to 1
 
 ::MODIFY GRADIENT WITH CODE::
 gradient.getMarks().clear();
 gradient.addMark(0.0f, ImColor(0.2f, 0.1f, 0.0f));
 gradient.addMark(0.7f, ImColor(120, 200, 255));
 
 ::WOOD BROWNS PRESET::
 gradient.getMarks().clear();
 gradient.addMark(0.0f, ImColor(0xA0, 0x79, 0x3D));
 gradient.addMark(0.2f, ImColor(0xAA, 0x83, 0x47));
 gradient.addMark(0.3f, ImColor(0xB4, 0x8D, 0x51));
 gradient.addMark(0.4f, ImColor(0xBE, 0x97, 0x5B));
 gradient.addMark(0.6f, ImColor(0xC8, 0xA1, 0x65));
 gradient.addMark(0.7f, ImColor(0xD2, 0xAB, 0x6F));
 gradient.addMark(0.8f, ImColor(0xDC, 0xB5, 0x79));
 gradient.addMark(1.0f, ImColor(0xE6, 0xBF, 0x83));
 
 */

#pragma once

#include "imgui.h"

#include <list>

struct ImGradientMark
{
    float color[4];
    float position; //0 to 1
};

class ImGradient
{
public:
    ImGradient();
    ~ImGradient();
    
    void getColorAt(float position, float* color) const;
    void addMark(float position, ImColor const color);
    void removeMark(ImGradientMark* mark);
    void refreshCache();
    std::list<ImGradientMark*> & getMarks(){ return m_marks; }
private:
    void computeColorAt(float position, float* color) const;
    std::list<ImGradientMark*> m_marks;
    float m_cachedValues[256 * 3];
};

namespace ImGui
{
    bool GradientButton(ImGradient* gradient);
    
    bool GradientEditor(ImGradient* gradient,
                        ImGradientMark* & draggingMark,
                        ImGradientMark* & selectedMark);
    

}
