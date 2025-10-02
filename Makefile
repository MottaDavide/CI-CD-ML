CONDA_ENV = drug_cicd

install:
	conda env update --file env.yaml --prune || conda env create --file env.yaml

format:
	black *.py

train:
	conda run -n $(CONDA_ENV) python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./results/model_results.png)' >> report.md
	conda run -n $(CONDA_ENV) cml comment create report.md

clean:
	conda env remove -n $(CONDA_ENV) -y

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	git fetch origin update
	git switch update
	git reset --hard origin/update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload davidemotta/CI-CD-ML ./app  /src --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload davidemotta/CI-CD-ML ./model /src/Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload davidemotta/CI-CD-ML ./results /src/Metrics --repo-type=space --commit-message="Sync Model"

deploy:	hf-login push-hub
