// https://gist.github.com/galloscript/8a5d179e432e062550972afcd1ecf112

//
//  imgui_color_gradient.cpp
//  imgui extension
//
//  Created by David Gallardo on 11/06/16.


#include "imgui_color_gradient.h"
#include "imgui_internal.h"

static const float GRADIENT_BAR_WIDGET_HEIGHT = 25;
static const float GRADIENT_BAR_EDITOR_HEIGHT = 40;
static const float GRADIENT_MARK_DELETE_DIFFY = -50;

ImGradient::ImGradient()
{
    addMark(0.0f, ImColor(0.0f,0.0f,0.0f));
    addMark(1.0f, ImColor(1.0f,1.0f,1.0f));
}

ImGradient::~ImGradient()
{
	for (ImGradientMark* mark : m_marks)
	{
		delete mark;
	}
}

void ImGradient::addMark(float position, ImColor const color)
{
    position = ImClamp(position, 0.0f, 1.0f);
	ImGradientMark* newMark = new ImGradientMark();
    newMark->position = position;
    newMark->color[0] = color.Value.x;
    newMark->color[1] = color.Value.y;
    newMark->color[2] = color.Value.z;
    
    m_marks.push_back(newMark);
    
    refreshCache();
}

void ImGradient::removeMark(ImGradientMark* mark)
{
    m_marks.remove(mark);
    refreshCache();
}

void ImGradient::getColorAt(float position, float* color) const
{
    position = ImClamp(position, 0.0f, 1.0f);    
    int cachePos = (position * 255);
    cachePos *= 3;
    color[0] = m_cachedValues[cachePos+0];
    color[1] = m_cachedValues[cachePos+1];
    color[2] = m_cachedValues[cachePos+2];
}

void ImGradient::computeColorAt(float position, float* color) const
{
    position = ImClamp(position, 0.0f, 1.0f);
    
    ImGradientMark* lower = nullptr;
    ImGradientMark* upper = nullptr;
    
    for(ImGradientMark* mark : m_marks)
    {
        if(mark->position < position)
        {
            if(!lower || lower->position < mark->position)
            {
                lower = mark;
            }
        }
        
        if(mark->position >= position)
        {
            if(!upper || upper->position > mark->position)
            {
                upper = mark;
            }
        }
    }
    
    if(upper && !lower)
    {
        lower = upper;
    }
    else if(!upper && lower)
    {
        upper = lower;
    }
    else if(!lower && !upper)
    {
        color[0] = color[1] = color[2] = 0;
        return;
    }
    
    if(upper == lower)
    {
        color[0] = upper->color[0];
        color[1] = upper->color[1];
        color[2] = upper->color[2];
    }
    else
    {
        float distance = upper->position - lower->position;
        float delta = (position - lower->position) / distance;
        
        //lerp
        color[0] = ((1.0f - delta) * lower->color[0]) + ((delta) * upper->color[0]);
        color[1] = ((1.0f - delta) * lower->color[1]) + ((delta) * upper->color[1]);
        color[2] = ((1.0f - delta) * lower->color[2]) + ((delta) * upper->color[2]);
    }
}

void ImGradient::refreshCache()
{
    m_marks.sort([](const ImGradientMark * a, const ImGradientMark * b) { return a->position < b->position; });
    
    for(int i = 0; i < 256; ++i)
    {
        computeColorAt(i/255.0f, &m_cachedValues[i*3]);
    }
}



namespace ImGui
{
    static void DrawGradientBar(ImGradient* gradient,
                                struct ImVec2 const & bar_pos,
                                float maxWidth,
                                float height)
    {
        ImVec4 colorA = {1,1,1,1};
        ImVec4 colorB = {1,1,1,1};
        float prevX = bar_pos.x;
        float barBottom = bar_pos.y + height;
        ImGradientMark* prevMark = nullptr;
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        draw_list->AddRectFilled(ImVec2(bar_pos.x - 2, bar_pos.y - 2),
                                 ImVec2(bar_pos.x + maxWidth + 2, barBottom + 2),
                                 IM_COL32(100, 100, 100, 255));
        
        if(gradient->getMarks().size() == 0)
        {
            draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                     ImVec2(bar_pos.x + maxWidth, barBottom),
                                     IM_COL32(255, 255, 255, 255));
            
        }
        
        ImU32 colorAU32 = 0;
        ImU32 colorBU32 = 0;
        
        for(auto markIt = gradient->getMarks().begin(); markIt != gradient->getMarks().end(); ++markIt )
        {
            ImGradientMark* mark = *markIt;
            
            float from = prevX;
            float to = prevX = bar_pos.x + mark->position * maxWidth;
            
            if(prevMark == nullptr)
            {
                colorA.x = mark->color[0];
                colorA.y = mark->color[1];
                colorA.z = mark->color[2];
            }
            else
            {
                colorA.x = prevMark->color[0];
                colorA.y = prevMark->color[1];
                colorA.z = prevMark->color[2];
            }
            
            colorB.x = mark->color[0];
            colorB.y = mark->color[1];
            colorB.z = mark->color[2];
            
            colorAU32 = ImGui::ColorConvertFloat4ToU32(colorA);
            colorBU32 = ImGui::ColorConvertFloat4ToU32(colorB);
            
            if(mark->position > 0.0)
            {
                
                draw_list->AddRectFilledMultiColor(ImVec2(from, bar_pos.y),
                                                   ImVec2(to, barBottom),
                                                   colorAU32, colorBU32, colorBU32, colorAU32);
            }
            
            prevMark = mark;
        }
        
        if(prevMark && prevMark->position < 1.0)
        {
            
            draw_list->AddRectFilledMultiColor(ImVec2(prevX, bar_pos.y),
                                               ImVec2(bar_pos.x + maxWidth, barBottom),
                                               colorBU32, colorBU32, colorBU32, colorBU32);
        }
        
        ImGui::SetCursorScreenPos(ImVec2(bar_pos.x, bar_pos.y + height + 10.0f));
    }
    
    static void DrawGradientMarks(ImGradient* gradient,
                                  ImGradientMark* & draggingMark,
                                  ImGradientMark* & selectedMark,
                                  struct ImVec2 const & bar_pos,
                                  float maxWidth,
                                  float height)
    {
        ImVec4 colorA = {1,1,1,1};
        ImVec4 colorB = {1,1,1,1};
        float barBottom = bar_pos.y + height;
        ImGradientMark* prevMark = nullptr;
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImU32 colorAU32 = 0;
        ImU32 colorBU32 = 0;
        
        for(auto markIt = gradient->getMarks().begin(); markIt != gradient->getMarks().end(); ++markIt )
        {
            ImGradientMark* mark = *markIt;
            
            if(!selectedMark)
            {
                selectedMark = mark;
            }
            
            float to = bar_pos.x + mark->position * maxWidth;
            
            if(prevMark == nullptr)
            {
                colorA.x = mark->color[0];
                colorA.y = mark->color[1];
                colorA.z = mark->color[2];
            }
            else
            {
                colorA.x = prevMark->color[0];
                colorA.y = prevMark->color[1];
                colorA.z = prevMark->color[2];
            }
            
            colorB.x = mark->color[0];
            colorB.y = mark->color[1];
            colorB.z = mark->color[2];
            
            colorAU32 = ImGui::ColorConvertFloat4ToU32(colorA);
            colorBU32 = ImGui::ColorConvertFloat4ToU32(colorB);
            
            draw_list->AddTriangleFilled(ImVec2(to, bar_pos.y + (height - 6)),
                                         ImVec2(to - 6, barBottom),
                                         ImVec2(to + 6, barBottom), IM_COL32(100, 100, 100, 255));
            
            draw_list->AddRectFilled(ImVec2(to - 6, barBottom),
                                     ImVec2(to + 6, bar_pos.y + (height + 12)),
                                     IM_COL32(100, 100, 100, 255), 1.0f, 1.0f);
            
            draw_list->AddRectFilled(ImVec2(to - 5, bar_pos.y + (height + 1)),
                                     ImVec2(to + 5, bar_pos.y + (height + 11)),
                                     IM_COL32(0, 0, 0, 255), 1.0f, 1.0f);
            
            if(selectedMark == mark)
            {
                draw_list->AddTriangleFilled(ImVec2(to, bar_pos.y + (height - 3)),
                                             ImVec2(to - 4, barBottom + 1),
                                             ImVec2(to + 4, barBottom + 1), IM_COL32(0, 255, 0, 255));
                
                draw_list->AddRect(ImVec2(to - 5, bar_pos.y + (height + 1)),
                                   ImVec2(to + 5, bar_pos.y + (height + 11)),
                                   IM_COL32(0, 255, 0, 255), 1.0f, 1.0f);
            }
            
            draw_list->AddRectFilledMultiColor(ImVec2(to - 3, bar_pos.y + (height + 3)),
                                               ImVec2(to + 3, bar_pos.y + (height + 9)),
                                               colorBU32, colorBU32, colorBU32, colorBU32);
            
            ImGui::SetCursorScreenPos(ImVec2(to - 6, barBottom));
            ImGui::InvisibleButton("mark", ImVec2(12, 12));
            
            if(ImGui::IsItemHovered())
            {
                if(ImGui::IsMouseClicked(0))
                {
                    selectedMark = mark;
                    draggingMark = mark;
                }
            }
            
            
            prevMark = mark;
        }
        
        ImGui::SetCursorScreenPos(ImVec2(bar_pos.x, bar_pos.y + height + 20.0f));
    }
    
    bool GradientButton(ImGradient* gradient)
    {
        if(!gradient) return false;
        
        ImVec2 widget_pos = ImGui::GetCursorScreenPos();
        // ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        float maxWidth = ImMax(250.0f, ImGui::GetContentRegionAvailWidth() - 100.0f);
        bool clicked = ImGui::InvisibleButton("gradient_bar", ImVec2(maxWidth, GRADIENT_BAR_WIDGET_HEIGHT));
        
        DrawGradientBar(gradient, widget_pos, maxWidth, GRADIENT_BAR_WIDGET_HEIGHT);
        
        return clicked;
    }
    
    bool GradientEditor(ImGradient* gradient,
                        ImGradientMark* & draggingMark,
                        ImGradientMark* & selectedMark)
    {
        if(!gradient) return false;
        
        bool modified = false;
        
        ImVec2 bar_pos = ImGui::GetCursorScreenPos();
        bar_pos.x += 10;
        float maxWidth = ImGui::GetContentRegionAvailWidth() - 20;
        float barBottom = bar_pos.y + GRADIENT_BAR_EDITOR_HEIGHT;
        
        ImGui::InvisibleButton("gradient_editor_bar", ImVec2(maxWidth, GRADIENT_BAR_EDITOR_HEIGHT));
        
        if(ImGui::IsItemHovered() && ImGui::IsMouseClicked(0))
        {
            float pos = (ImGui::GetIO().MousePos.x - bar_pos.x) / maxWidth;
            
            float newMarkCol[4];
            gradient->getColorAt(pos, newMarkCol);
            

            gradient->addMark(pos, ImColor(newMarkCol[0], newMarkCol[1], newMarkCol[2]));
        }
        
        DrawGradientBar(gradient, bar_pos, maxWidth, GRADIENT_BAR_EDITOR_HEIGHT);
        DrawGradientMarks(gradient, draggingMark, selectedMark, bar_pos, maxWidth, GRADIENT_BAR_EDITOR_HEIGHT);
        
        if(!ImGui::IsMouseDown(0) && draggingMark)
        {
            draggingMark = nullptr;
        }
        
        if(ImGui::IsMouseDragging(0) && draggingMark)
        {
            float increment = ImGui::GetIO().MouseDelta.x / maxWidth;
            bool insideZone = (ImGui::GetIO().MousePos.x > bar_pos.x) &&
                              (ImGui::GetIO().MousePos.x < bar_pos.x + maxWidth);
            
            if(increment != 0.0f && insideZone)
            {
                draggingMark->position += increment;
                draggingMark->position = ImClamp(draggingMark->position, 0.0f, 1.0f);
                gradient->refreshCache();
                modified = true;
            }
            
            float diffY = ImGui::GetIO().MousePos.y - barBottom;
            
            if(diffY <= GRADIENT_MARK_DELETE_DIFFY)
            {
                gradient->removeMark(draggingMark);
                draggingMark = nullptr;
                selectedMark = nullptr;
                modified = true;
            }
        }
        
        if(!selectedMark && gradient->getMarks().size() > 0)
        {
            selectedMark = gradient->getMarks().front();
        }
        
        if(selectedMark)
        {
            bool colorModified = ImGui::ColorPicker3("color", selectedMark->color);
            
            if(selectedMark && colorModified)
            {
                modified = true;
                gradient->refreshCache();
            }
        }
        
        return modified;
    }
};
