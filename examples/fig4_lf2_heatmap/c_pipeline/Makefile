CC      = gcc
CFLAGS  = -O3 -march=native -std=c11 -Wall -Wextra
TARGET  = lf2_solver.exe
SRC     = lf2_solver.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGET)
