
TRACK_DIR = /Users/chrisjr/Desktop/tracks

ROOT = /Users/chrisjr/Development/susurrant_prep
VOCAB_DIR = $(ROOT)/vocab
TRACK_FILE = $(ROOT)/tracks.h5

WHOLE_TOKEN_FILE = $(VOCAB_DIR)/tokens.h5
SEGMENTED_TOKEN_FILE = $(ROOT)/segmented.h5

INVERSE_DICT = $(VOCAB_DIR)/vocab.json

UTIL_EXE = $(ROOT)/susurrant-utils/susurrant

LDA_DIR = $(ROOT)/vw
LDA_IN = $(LDA_DIR)/data.vw
LDA_OUT = $(LDA_DIR)/topics.dat

VIZ_DIR = $(ROOT)/susurrant_elm/data
VIZ_METADATA = $(VIZ_DIR)/doc_metadata.json
VIZ_TOPICS = $(VIZ_DIR)/topics.json

RM ?= rm -f
PYTHON ?= python

DTYPES = beat_coefs chroma gfccs
KMEANS_INPUT = $(addsuffix .h5, $(addprefix $(VOCAB_DIR)/train/,$(DTYPES)))

KMEANS_TYPES = $(addprefix $(VOCAB_DIR)/train/clusters_,$(DTYPES))
KMEANS_RESULT = $(addsuffix .txt,$(KMEANS_TYPES))
ANN_FILES = $(addsuffix .tree,$(KMEANS_TYPES))

.PHONY: clean
.PHONY: all
.PHONY: track_data
.PHONY: .viz_data

all: $(SEGMENTED_TOKEN_FILE) $(INVERSE_DICT) .viz_data

clean:
	$(RM) $(VOCAB_DIR)/train/*.tree

.viz_data: $(VIZ_METADATA) $(VIZ_TOPICS) track_data

# Analyze audio
$(TRACK_FILE): $(TRACK_DIR)/*.mp3
	$(PYTHON) proc_all_files.py $(TRACK_DIR) $(TRACK_FILE)

# Split analysis by type (GFCC, chroma, etc.)
$(KMEANS_INPUT): $(TRACK_FILE)
	$(PYTHON) track_to_kmean_training.py $< $@

# Take samples from larger token list
%_sampled.h5: %.h5
	$(PYTHON) downsample.py $< $@

# Train k-means
clusters_%.txt: %_sampled.h5
	$(PYTHON) sofia_kmeans.py $< $@

# Train ANNs
%.tree: %.txt
	$(PYTHON) index_clusters.py $< $@

# Tracks to k-means assignments
$(WHOLE_TOKEN_FILE): $(ANN_FILES)
	$(PYTHON) tracks_to_assignments.py $(TRACK_FILE) $@

# Token file to segmented token file
$(SEGMENTED_TOKEN_FILE): $(WHOLE_TOKEN_FILE)
	$(PYTHON) segment.py $< $@
	touch $@

$(INVERSE_DICT): $(ANN_FILES)
	$(PYTHON) gen_inverses.py $(INVERSE_DICT)

$(LDA_IN): $(SEGMENTED_TOKEN_FILE)
	$(UTIL_EXE) to_vw -i $(SEGMENTED_TOKEN_FILE) -o $(LDA_IN)

$(LDA_OUT): $(LDA_IN)
	$(PYTHON) run_vw.py $< $(TOPICS)

$(VIZ_METADATA):
	$(PYTHON) gen_json.py metadata $(VIZ_DIR)

$(VIZ_TOPICS): $(LDA_OUT)
	$(PYTHON) gen_json.py vw $(VIZ_DIR) $(LDA_DIR)

track_data: $(SEGMENTED_TOKEN_FILE)
	$(PYTHON) gen_json.py tracks $(VIZ_DIR) $(SEGMENTED_TOKEN_FILE)