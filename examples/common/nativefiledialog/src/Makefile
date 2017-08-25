all:
	mkdir ../bin
	g++ -O3 -s -o ../bin/testopen -I"include/" ../test/test_opendialog.c nfd_win.cpp nfd_common.c -O2 -s -march=native -funroll-loops -mfpmath=sse -msse2 -msse3 -mssse3 -ffast-math    -Wall -Wunused -Wno-format-y2k  -fno-exceptions -fno-strict-aliasing   -m64 -static  -static-libgcc -static-libstdc++  -lole32 -luuid -lcomctl32 
	g++ -O3 -s -o ../bin/testopen-mul -I"include/" ../test/test_opendialogmultiple.c nfd_win.cpp nfd_common.c -O2 -s -march=native -funroll-loops -mfpmath=sse -msse2 -msse3 -mssse3 -ffast-math    -Wall -Wunused -Wno-format-y2k  -fno-exceptions -fno-strict-aliasing   -m64 -static  -static-libgcc -static-libstdc++  -lole32 -luuid -lcomctl32 
	g++ -O3 -s -o ../bin/test_pickfolder -I"include/" ../test/test_pickfolder.c nfd_win.cpp nfd_common.c -O2 -s -march=native -funroll-loops -mfpmath=sse -msse2 -msse3 -mssse3 -ffast-math    -Wall -Wunused -Wno-format-y2k  -fno-exceptions -fno-strict-aliasing   -m64 -static  -static-libgcc -static-libstdc++  -lole32 -luuid -lcomctl32 
	g++ -O3 -s -o ../bin/test_savedialog -I"include/" ../test/test_savedialog.c nfd_win.cpp nfd_common.c -O2 -s -march=native -funroll-loops -mfpmath=sse -msse2 -msse3 -mssse3 -ffast-math    -Wall -Wunused -Wno-format-y2k  -fno-exceptions -fno-strict-aliasing   -m64 -static  -static-libgcc -static-libstdc++  -lole32 -luuid -lcomctl32 
	
clean:
	rm -rf ../bin