rmdir /s /q build
mkdir build

cmake.exe -G "Visual Studio 17 2022" -A x64 -B build -S .
