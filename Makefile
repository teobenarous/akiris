DATA_ASSETS := data/assets
RAW_DIR := data/raw
SAMPLE_DIR := data/sample
POP_SIZE := 25000

.PHONY: all generate-train split-data generate-stream clean

all: clean generate-train split-data generate-stream

$(RAW_DIR) $(SAMPLE_DIR):
	@mkdir -p $@

generate-train: | $(RAW_DIR)
	@echo "Generating Training Data (2 Years)..."
	python scripts/generator/generator.py \
		--data $(DATA_ASSETS) \
		--output $(RAW_DIR) \
		--days 730 \
		--history 90 \
		--mean_daily_admits 20 \
		--population_size $(POP_SIZE)

split-data: generate-train | $(SAMPLE_DIR)
	@echo "Splitting into Stratified Train/Test Sets..."
	python scripts/generator/split_train_test.py \
		--input $(RAW_DIR)/dataset.csv \
		--train-out $(SAMPLE_DIR)/train.csv \
		--test-out $(SAMPLE_DIR)/test.csv \
		--test-size 0.2

generate-stream: | $(SAMPLE_DIR)
	@echo "Generating Live Showcase Stream (90 days)..."
	python scripts/generator/generator.py \
		--data $(DATA_ASSETS) \
		--output $(SAMPLE_DIR) \
		--days 90 \
		--history 90 \
		--mean_daily_admits 20 \
		--population_size $(POP_SIZE)
	@echo "Cleaning up raw artifacts..."
	@rm -rf $(RAW_DIR) && rm -f $(SAMPLE_DIR)/dataset.csv
	@echo "Pipeline Complete. Assets saved to $(SAMPLE_DIR)/"

clean:
	@echo "Wiping existing data..."
	rm -rf $(RAW_DIR) $(SAMPLE_DIR)