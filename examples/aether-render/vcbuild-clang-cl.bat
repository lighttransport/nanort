rmdir /q /s build
cmake.exe -T LLVM-vs2014 -G "Visual Studio 15 2017 Win64" -Bbuild -H. -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/msbuild-bin/cl.exe" -DBOOST_ROOT=C:\local\boost_1_65_1 -DEIGEN3_ROOT=C:\local\eigen-3
