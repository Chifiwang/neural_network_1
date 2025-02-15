CC = clang++
CFLAGS = -g -Wall -Wextra

all: main

main: main.o layer.o neural_network.o matrix.o
	${CC} ${CFLAGS} -o $@ $^

%.o: %.c
	${CC} ${CFLAGS} -c $<

clean:
	rm *.o main
