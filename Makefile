DATA_DIR=./data
DATA_DIR_JNS=$(DATA_DIR)/journals # journal data
DATA_DIR_ANNOTS=$(DATA_DIR)/annotated # annotation data
DATA_DIR_PROCESSED=$(DATA_DIR)/processed # processed data
DATA_DIR_KSPLIT=$(DATA_DIR)/DATA_DIR_KSPLIT # trainin data

SRC_DIR=./src

# run once to create the dark data on label studio
create-dark-data-LS:
	cd src && python -c "from label_studio import labelStudio; LS = labelStudio(); LS.create_project(LS.dark_data_project)"

# Scrape google scholar to control for different field of studies
import-gscholar-jns:
	python $(SRC_DIR)/import/scrape_gscholar.py -o $(DATA_DIR_JNS)

# We create a collection in mongoDB where each document is 
# a section of papers from selected venues.
create-s2orc-dark-data-collection:
	python $(SRC_DIR)/import/create_s2orc_dark_data_collection.py -i $(DATA_DIR_JNS) 


more-data-annotations:
	cd src && python -c "from label_studio import labelStudio; LS = labelStudio(); LS.dispatch_annots(keyword='data')"

more-code-annotations:
	cd src && python -c "from label_studio import labelStudio; LS = labelStudio(); LS.dispatch_annots(keyword='code')"

more-software-annotations:
	cd src && python -c "from label_studio import labelStudio; LS = labelStudio(); LS.dispatch_annots(keyword='software')"

# We train-test split the data, somewhere. I don't remember where...
create-train-test:
	cd src && python $(SRC_DIR)/train_test.py

fine-tune-llama3:
	tune run --nproc_per_node 4 lora_finetune_distributed --config ./my_custom_config.yaml dtype=fp32

# Next 3 steps are now obsolete because we use label studio. 

# We use clusteredDataStatement as main file for the annotations.
# We lost track of the script producing the data file...
# cluster-data:
# 	python $(SRC_DIR)/process/cluster_data.py -i $(DATA_DIR_JNS) -o $(DATA_DIR_PROCESSED)

# original round of annotations using excel sheet
# assign-data-round-1:
# 	python $(SRC_DIR)/annotations/2023-03-07-assign-data.py -i $(DATA_DIR_PROCESSED) -o $(DATA_DIR_ANNOTS)

# assign-data-round-2:
# 	python $(SRC_DIR)/import/2023-09-21-assign-data.py -i $(processed) -o $(DATA_DIR_ANNOTS)
