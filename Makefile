PYTHONPATH := src
export PYTHONPATH

.PHONY: smoke names_v3 multipref

# Toy end-to-end check, a few minutes on one GPU.
smoke:
	python src/experiment_multipref.py all --config configs/smoke_names.yaml

names_v3:
	python src/experiment_multipref.py generate --config configs/names_v3.yaml
	bash scripts/queue.sh configs/names_v3.yaml

multipref:
	python src/experiment_multipref.py generate --config configs/multipref.yaml
	bash scripts/queue.sh configs/multipref.yaml
