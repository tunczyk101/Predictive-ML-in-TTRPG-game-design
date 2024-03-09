tests:
	cd ./test && pytest .

test_file:
	cd ./test && pytest $(FILE)

install:
	poetry lock && poetry install --sync
	poetry export --without-hashes --format=requirements.txt --output=requirements.txt

# make install or MinGW32-make install
