install:
    conda env update --file env.yaml --prune || conda env create --file env.yaml
    conda run -n drug_cicd pip install --upgrade pip

format:
    conda run -n drug_cicd black *.py

train:
    conda run -n drug_cicd python train.py

eval:
    echo "## Model Metrics" > report.md
    cat ./results/metrics.txt >> report.md
   
    echo '\n## Confusion Matrix Plot' >> report.md
    echo '![Confusion Matrix](./results/model_results.png)' >> report.md
   
    conda run -n drug_cicd cml comment create report.md

clean:
	conda env remove -n drug_cicd -y