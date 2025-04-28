CC = clang++
CFLAGS = -g -Wall -Wextra -O3

all: utils.hpp random.hpp main

main: main.o neural_network.o
	${CC} ${CFLAGS} -o $@ $^

%.o: %.cpp %.hpp
	${CC} ${CFLAGS} -c $<

.PHONY: clean
clean:
	rm *.o main
