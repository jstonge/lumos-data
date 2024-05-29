DATA_DIR=./data
DATA_DIR_JNS=$(DATA_DIR)/journals # journal data
DATA_DIR_ANNOTS=$(DATA_DIR)/annotated # annotation data
DATA_DIR_PROCESSED=$(DATA_DIR)/processed # processed data
DATA_DIR_KSPLIT=$(DATA_DIR)/DATA_DIR_KSPLIT # trainin data

SRC_DIR=./src

# Scrape google scholar to control for different field of studies
import-gscholar-jns:
	python $(SRC_DIR)/import/scrape_gscholar.py -o $(DATA_DIR_JNS)

# We create a collection in mongoDB where each document is 
# a section containing a mention of data. The sections come from
# papers from selected journals that were in the s2orc database.
create-s2orc-dark-data-collection:
	python $(SRC_DIR)/import/create_s2orc_dark_data_collection.py -i $(DATA_DIR_JNS) 

# We use clusteredDataStatement as main file for the annotations
# I don't know where the data comes from, but it is in the data folder...
cluster-data:
	python $(SRC_DIR)/process/cluster_data.py -i $(DATA_DIR_JNS) -o $(DATA_DIR_PROCESSED)

assign-data-round-1:
	python $(SRC_DIR)/annotations/2023-03-07-assign-data.py -i $(DATA_DIR_PROCESSED) -o $(DATA_DIR_ANNOTS)
	
assign-data-round-2:
	python $(SRC_DIR)/import/2023-09-21-assign-data.py -i $(processed) -o $(DATA_DIR_ANNOTS)

# We train-test split the data, somewhere. I don't remember where...
create-train-test:
	python $(SRC_DIR)/ksplit/train_test.py -i $(DATA_DIR_ANNOTS) -o $(DATA_DIR_KSPLIT)