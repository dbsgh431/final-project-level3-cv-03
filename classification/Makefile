run_black:
	python -m black . -l 119

run_server:
	python -m app

run_client:
	python -m streamlit run app/frontend.py --server.port 30018

run_app: run_server run_client

run_assignment_tests:
	poetry run pytest assignments/app_test.py