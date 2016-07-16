# stb_iamge_write.h is not C89 compliant, so compile it separately.
all:
	clang -c -g -std=c99 imageio.c
	clang -o test -Weverything -g -O0 -ansi -pedantic test.c imageio.o -lm

clean:
	rm imageio.o test 
