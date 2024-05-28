DATA_DIR=./data
DATA_DIR_JNS=$(DATA_DIR)/journals
DATA_DIR_ANNOTS=$(DATA_DIR)/annotated


SRC_DIR=./src

MODEL_DIR=$(SCRIPT_DIR)/models


#####################
#                   #
#       GLOBAL      #
#                   #
#####################

clean:
	rm -rf docs/.observablehq/cache

###################################
#                                 #
#        More annotated data      #
#                                 #
###################################

import-gscholar-jns:
	python $(SRC_DIR)/import/scrape_gscholar.py -o $(DATA_DIR_JNS)