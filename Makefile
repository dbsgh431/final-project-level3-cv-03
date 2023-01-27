run_black:
	python -m black . -l 119

run_server:
	python -m app/main.py

run_client:
	python -m streamlit run app/frontend.py --server.port 30006 --server.fileWatcherType None

run_app: run_server run_client

run_assignment_tests:
	poetry run pytest assignments/app_test.py