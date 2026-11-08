SRC := src/main.s src/load.s src/tok.s src/model.s src/math.s src/par.s src/sme.s
OBJ := $(SRC:src/%.s=build/%.o)
SDK := $(shell xcrun --show-sdk-path)
HF  := https://huggingface.co/openai-community/gpt2/resolve/main

gpt2: $(OBJ)
	ld -arch arm64 -e _main -o $@ $(OBJ) -L$(SDK)/usr/lib -lSystem

build/%.o: src/%.s src/macros.inc
	@mkdir -p build
	as -arch arm64 -I src $< -o $@

# The original Hugging Face files. No conversion: gpt2 reads them as they are.
weights: model.safetensors vocab.json merges.txt
model.safetensors vocab.json merges.txt:
	curl -fL -o $@ $(HF)/$@

clean:
	rm -rf build gpt2

.PHONY: weights clean
