TRACK_DIR = $(HOME)/Desktop/tracks

ROOT = $(abspath $(CURDIR)/..)
VOCAB_DIR = $(ROOT)/vocab
TRACK_FILE = $(ROOT)/tracks.h5

WHOLE_TOKEN_FILE = $(VOCAB_DIR)/tokens.h5
SEGMENTED_TOKEN_FILE = $(ROOT)/segmented.h5
COMMENT_FILE = $(ROOT)/parsed_comments.json

MALLET_BIN = $(HOME)/Applications/mallet-2.0.8RC2/bin/mallet

UTIL_EXE = $(ROOT)/susurrant-utils/susurrant

LDA_TYPE = MALLET

LDA_DIR = $($(LDA_TYPE)_DIR)
LDA_IN = $($(LDA_TYPE)_IN)
LDA_OUT = $($(LDA_TYPE)_OUT)

# INCLUDE_COMMENTS = --text-in $(COMMENT_FILE)
INCLUDE_COMMENTS =

VW_DIR = $(ROOT)/vw
VW_IN = $(VW_DIR)/data.vw
VW_OUT = $(VW_DIR)/topics.dat

MALLET_DIR = $(ROOT)/mallet

MALLET_USE_BULK_LOADER = false
MALLET_USE_SCALA = true
MALLET_DO_PRUNE = true

MALLET_IN_TEXT = $(MALLET_DIR)/instances.txt

ifeq ($(MALLET_DO_PRUNE),true)
MALLET_IN = $(MALLET_IN_PRUNED)
else
MALLET_IN = $(MALLET_IN_RAW)
endif

MALLET_IN_RAW = $(MALLET_DIR)/instances.mallet
MALLET_IN_PRUNED = $(MALLET_DIR)/instances_pruned.mallet

MALLET_OUT = $(MALLET_DIR)/topic-state.gz

VIZ_DIR = $(ROOT)/susurrant_elm/data
VOCAB_DICT = $(VIZ_DIR)/vocab.json
INVERSE_DICT = $(VIZ_DIR)/inverses.json
VIZ_METADATA = $(VIZ_DIR)/doc_metadata.json
VIZ_TOPICS = $(VIZ_DIR)/topics.json

TOPICS = 10

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

vw: $(VW_IN) $(VW_OUT)

mallet: $(MALLET_IN) $(MALLET_OUT)

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

$(COMMENT_FILE):
	$(PYTHON) extract_comments.py $@

$(VOCAB_DICT): $(ANN_FILES)
	$(PYTHON) gen_inverses.py vocab $@

$(INVERSE_DICT): $(ANN_FILES)
	$(PYTHON) gen_inverses.py invert $@

$(VW_IN): $(SEGMENTED_TOKEN_FILE) $(COMMENT_FILE)
	$(UTIL_EXE) to_vw -i $(SEGMENTED_TOKEN_FILE) -o $@ $(INCLUDE_COMMENTS)

$(VW_OUT): $(VW_IN)
	$(PYTHON) run_vw.py $< $(TOPICS)

$(MALLET_IN_TEXT): $(SEGMENTED_TOKEN_FILE) $(COMMENT_FILE)
	$(UTIL_EXE) to_mallet_text -i $(SEGMENTED_TOKEN_FILE) -o $@ $(INCLUDE_COMMENTS)

ifeq ($(MALLET_USE_BULK_LOADER),true)
$(MALLET_IN_RAW): $(MALLET_IN_TEXT)
	$(MALLET_BIN) run cc.mallet.util.BulkLoader \
		--keep-sequence true \
		--input $< \
		--prune-count 3 \
		--output $@
else
$(MALLET_IN_RAW): $(SEGMENTED_TOKEN_FILE) $(COMMENT_FILE)
	$(UTIL_EXE) to_mallet -i $(SEGMENTED_TOKEN_FILE) -o $@ $(INCLUDE_COMMENTS)
endif

$(MALLET_IN_PRUNED): $(MALLET_IN_RAW)
	$(UTIL_EXE) prune_mallet -i $(MALLET_IN_RAW) --min-doc-freq 10 --num-words 500

ifeq ($(MALLET_USE_SCALA),true)
$(MALLET_OUT): $(MALLET_IN)
	$(UTIL_EXE) train_mallet -i $< --topics $(TOPICS)
else
$(MALLET_OUT): $(MALLET_IN)
	$(PYTHON) run_lda.py $< $(MALLET_DIR) $(TOPICS)
endif

$(VIZ_METADATA):
	$(PYTHON) gen_json.py metadata $(VIZ_DIR)

$(VIZ_TOPICS): $(LDA_OUT)
ifeq ($(LDA_TYPE),VW)
	$(PYTHON) gen_json.py vw $(VIZ_DIR) $(VW_DIR)
else
	$(PYTHON) gen_json.py mallet $(VIZ_DIR) $(MALLET_DIR)
endif

track_data: $(SEGMENTED_TOKEN_FILE)
#	$(PYTHON) gen_json.py tracks $(VIZ_DIR) $(SEGMENTED_TOKEN_FILE)
	$(UTIL_EXE) tracks -i $(WHOLE_TOKEN_FILE) -o $(VIZ_DIR)/tracks

$(AUDIO_FILE): $(ANN_FILES)
	$(PYTHON) gen_inverses.py audio_file $(AUDIO_FILE) $(AUDIO_INDEX)
