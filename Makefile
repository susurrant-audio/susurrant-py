TRACK_DIR = /Users/chrisjr/Desktop/tracks

ROOT = /Users/chrisjr/Development/susurrant_prep
VOCAB_DIR = $(ROOT)/vocab
TRACK_FILE = $(ROOT)/tracks.h5

WHOLE_TOKEN_FILE = $(VOCAB_DIR)/tokens.h5
SEGMENTED_TOKEN_FILE = $(ROOT)/segmented.h5

UTIL_EXE = $(ROOT)/susurrant-utils/susurrant

LDA_DIR = $(ROOT)/vw
LDA_IN = $(LDA_DIR)/data.vw
LDA_OUT = $(LDA_DIR)/topics.dat

VIZ_DIR = $(ROOT)/susurrant_elm/data
VOCAB_DICT = $(VIZ_DIR)/vocab.json
INVERSE_DICT = $(VIZ_DIR)/inverses.json
VIZ_METADATA = $(VIZ_DIR)/doc_metadata.json
VIZ_TOPICS = $(VIZ_DIR)/topics.json

AUDIO_FILE = $(ROOT)/susurrant-max/samples.wav
AUDIO_INDEX = $(ROOT)/susurrant-max/samples.json

RM ?= rm -f
PYTHON ?= python

DTYPES = beat_coefs chroma gfccs
KMEANS_INPUT = $(addsuffix .h5, $(addprefix $(VOCAB_DIR)/train/,$(DTYPES)))
KMEANS_DOWNSAMPLED = $(addsuffix _sampled.h5, $(addprefix $(VOCAB_DIR)/train/,$(DTYPES)))


ELKI_FILES = $(addsuffix .elki, $(addprefix $(VOCAB_DIR)/train/,$(DTYPES)))
DBSCAN_RESULTS = $(addsuffix .dbscan, $(addprefix $(VOCAB_DIR)/train/,$(DTYPES)))

KMEANS_TYPES = $(addprefix $(VOCAB_DIR)/train/clusters_,$(DTYPES))
KMEANS_RESULT = $(addsuffix .txt,$(KMEANS_TYPES))
ANN_FILES = $(addsuffix .tree,$(KMEANS_TYPES))

.PRECIOUS: %.h5 %.txt %.elki %_sampled.h5
.SECONDARY: $(KMEANS_RESULT) $(DBSCAN_RESULTS) $(KMEANS_DOWNSAMPLED)

.PHONY: clean
.PHONY: all
.PHONY: track_data
.PHONY: .viz_data
.PHONY: downsampled
.PHONY: elki


all: $(SEGMENTED_TOKEN_FILE) .viz_data

clean:
	$(RM) $(VOCAB_DIR)/train/*.tree

downsampled: $(KMEANS_DOWNSAMPLED)
elki: $(DBSCAN_RESULTS)

.viz_data: $(VIZ_METADATA) $(VIZ_TOPICS) $(VOCAB_DICT) # track_data

audio: $(AUDIO_FILE) $(AUDIO_INDEX)

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

# Convert to ELKI format
%.elki: %_sampled.h5
	$(UTIL_EXE) elki_prep -i $< -o $@

%.dbscan: %_sampled.h5
	$(PYTHON) run_dbscan.py $< $@

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

$(VOCAB_DICT): $(ANN_FILES)
	$(PYTHON) gen_inverses.py vocab $@

$(INVERSE_DICT): $(ANN_FILES)
	$(PYTHON) gen_inverses.py invert $@

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

$(AUDIO_FILE): $(ANN_FILES)
	$(PYTHON) gen_inverses.py audio_file $(AUDIO_FILE) $(AUDIO_INDEX)
