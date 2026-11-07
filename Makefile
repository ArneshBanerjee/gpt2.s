SRC := src/main.s src/tok.s src/model.s src/math.s
OBJ := $(SRC:src/%.s=build/%.o)
SDK := $(shell xcrun --show-sdk-path)

gpt2: $(OBJ)
	ld -arch arm64 -e _main -o $@ $(OBJ) -L$(SDK)/usr/lib -lSystem

build/%.o: src/%.s src/macros.inc
	@mkdir -p build
	as -arch arm64 -I src $< -o $@

weights: gpt2_124M.bin
gpt2_124M.bin gpt2_tok.bin:
	uv run python tools/export.py

clean:
	rm -rf build gpt2

.PHONY: weights clean
