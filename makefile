CC = clang++
CFLAGS = -g -Wall -Wextra

all: main

main: main.o matrix.o neural_network.o
	${CC} ${CFLAGS} -o $@ $^

%.o: %.c %.h
	${CC} ${CFLAGS} -c $<

clean:
	rm *.o main
