all:
	gcc main.c -o main -lm -lSDL2 -Wall -Wextra 

clean:
	rm -f main *.o