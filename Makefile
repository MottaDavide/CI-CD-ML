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
