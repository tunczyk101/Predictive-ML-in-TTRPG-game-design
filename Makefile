tests:
	cd ./test && pytest .

test_file:
	cd ./test && pytest $(FILE)

install:
	poetry lock && poetry install --sync

# make install or MinGW32-make install
