all: compile run

compile:
	@echo "Compiling..."
	@g++ -std=c++11 -o main main.cpp

run:
	@echo "Running..."
	@./main