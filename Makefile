CONDA_ENV = drug_cicd

install:
	conda env update --file env.yaml --prune || conda env create --file env.yaml

format:
	black *.py

test:
	conda run -n $(CONDA_ENV) pytest -q
	
coverage:
	conda run -n $(CONDA_ENV) pytest --cov=./ --cov-report=term --cov-report=xml --cov-fail-under=60

train:
	conda run -n $(CONDA_ENV) python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./results/model_results.png)' >> report.md
	conda run -n $(CONDA_ENV) cml comment create report.md

promote:
	conda run -n $(CONDA_ENV) python promote_model.py

clean:
	conda env remove -n $(CONDA_ENV) -y

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	hf auth login --token $(HF) --add-to-git-credential

push-hub:
	hf upload davidemotta/CI-CD-ML ./app  /src --repo-type=space --commit-message="Sync App files"
	hf upload davidemotta/CI-CD-ML ./model /model --repo-type=space --commit-message="Sync Model"
	hf upload davidemotta/CI-CD-ML ./results /metrics --repo-type=space --commit-message="Sync Model"
	hf upload davidemotta/CI-CD-ML ./app/requirements.txt /requirements.txt --repo-type=space --commit-message="Sync root requirements"

deploy:	hf-login push-hub
