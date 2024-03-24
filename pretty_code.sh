black .  --exclude="src/submodules/"
isort --profile black . -s src/submodules/
pipreqs . --force
pytest tests --cov-report term-missing --cov src/ -ra