SHELL:=/bin/bash
VENV=.venv
PYTHON_VERSION=3.10
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip
ACT="./bin/act"
MAKEFLAGS += --no-print-directory


create: requirements.txt
	@if ! [ -d $(VENV) ]; then\
        python${PYTHON_VERSION} -m venv $(VENV);\
		$(PIP) install --upgrade pip;\
		$(PIP) install -r requirements.txt;\
	fi

create-dev: create
	$(PIP) install pycodestyle==2.10.0
	$(PIP) install pylint==2.17.4 autopep8==2.0.2
	curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

update:
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

.PHONY: run clean


test-cicd-lint-local:
	${ACT} -j markdown-lint -W .github/workflows/doc.yml

test-cicd-code-local:
	${ACT} -j python-lint -W .github/workflows/code.yml

test-code:
	$(PYTHON) -m pycodestyle scripts
	$(PYTHON) -m pylint scripts

test-lint: test-cicd-lint-local

####################################################
################## TEST VARIABLES ##################
####################################################

IS_DEBUG=1
IS_MISS_JBSUB=$(shell which jbsub &>/dev/null && echo 0 || echo 1)

LOG_DIRNAME=./logs
DATASETS_DIRNAME=./datasets
AUTOML_DIRNAME=automl_search

GPUS=0
CORES=6
MEMORY=32
EPOCHS=1000
PATIENCE=20
TEST_SIZE=0.3
FINETUNE_SIZE=0.5
VALIDATION_SIZE=0.2
TIME_LIMIT=40000
AUTOML_TRIALS=1000
SUBSETS_TRIALS=1000
MODEL_PRUNING_TRIALS=10000
MODELS_PRUNED_FOR_SUBSETS_MAX=100
PERFORMANCE_DEGRADATION=0.10
EVAL_METRIC="accuracy"
MODEL_NAME="best_model"

PLOT_METRIC= "accuracy"
PROBLEM_TYPE= "binary"
COMPOSITION= "balanced"
RANKING_RECURSIVENESS= "recursive"
SUBSET_LEFT_TAKEN_POLICIY= "random"
FEATURE_RANKING_ALGORITHM= "custom_sbe"
SUBSETS_TARGET_DIR= ""

BEST_FEATURES_FOR_SUBSET_AMOUNTS=  0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00
MODEL_PRUNING_AMOUNTS=             0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00
DATASET= "sample"
SUBSETS_SIZE_AMOUNT= 0.5
MODEL_PRUNING_ALGORITHM= "globally_structured_connections_l1"

# Iterate over following variables

LIST_DATASET= "CICIDS2017" "ICS-D1-BIN" "EDGE2022"
LIST_SUBSETS_SIZE_AMOUNTS=                   0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
LIST_MODEL_PRUNING_ALGORITHMS= "globally_structured_connections_l1" "globally_structured_neurons_l1"


#######################################################
################## HELPER TO FORMAT ###################
#######################################################

create-dir:
    ifndef DIR_TO_CREATE
	$(error DIR_TO_CREATE is undefined)
    else
	@mkdir -p ${DIR_TO_CREATE}
    endif

get-cmd-formatted:
    ifndef CMD_NAME
	$(error CMD_NAME is undefined)
    endif
    ifndef DATASET
	$(error DATASET is undefined)
    endif
    ifeq (${IS_MISS_JBSUB}, 1)
    ifeq (${IS_DEBUG},1)
	@DATETIME=$$(date '+%Y-%m-%d-%H:%M:%S') && echo "echo ''@'2>&1 | tee ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${CMD_NAME}/$${DATETIME}.log'"
    else
	@${MAKE} DIR_TO_CREATE=${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${CMD_NAME} create-dir
	@DATETIME=$$(date '+%Y-%m-%d-%H:%M:%S') && echo ": && @2>&1 | tee ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${CMD_NAME}/$${DATETIME}.log"
    endif
    else
    ifeq (${IS_DEBUG},1)
	@DATETIME=$$(date '+%Y-%m-%d-%H:%M:%S') && echo echo "jbsub -cores 1x${CORES}+${GPUS} -mem ${MEMORY}G -out ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${CMD_NAME}/$${DATETIME}.jbinfo -err ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${CMD_NAME}/$${DATETIME}.log @ && :"
    else
	@${MAKE} DIR_TO_CREATE=${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${CMD_NAME} create-dir
	@DATETIME=$$(date '+%Y-%m-%d-%H:%M:%S') && echo "jbsub -cores 1x${CORES}+${GPUS} -mem ${MEMORY}G -out ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${CMD_NAME}/$${DATETIME}.jbinfo -err ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${CMD_NAME}/$${DATETIME}.log @ && :"
    endif
    endif

#######################################################
## RUN SINGLE EITHER LOCAL OR FROM JBSUB-ENABLED ENV ##
#######################################################

run-dataset-creator:
	@TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS} get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/dataset_creator.py \
		--directory ${DATASETS_DIRNAME}/${DATASET}/original\
		--test-size ${TEST_SIZE}\
		--finetune-size ${FINETUNE_SIZE}\
		--validation-size ${VALIDATION_SIZE}\
		--cpus ${CORES}\
		--composition ${COMPOSITION}\
		--problem-type ${PROBLEM_TYPE}\
		$${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-automl-search:
	@TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${TIME_LIMIT}t-${EVAL_METRIC}m-${AUTOML_TRIALS}a-${PATIENCE}p-${EPOCHS}e get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/automl_model_search.py search \
		--directory ${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}\
		--cpus ${CORES}\
		--gpus ${GPUS}\
		--attempts ${AUTOML_TRIALS}\
		--epochs ${EPOCHS}\
		--patience ${PATIENCE}\
		--time-limit ${TIME_LIMIT}\
		--metric ${EVAL_METRIC}\
		$${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-automl-dump:
	@TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME} get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/automl_model_search.py dump --directory ${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${AUTOML_DIRNAME}\
	                                                              --model-name ${MODEL_NAME}\
																  $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-feature-ranking:
	@TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${FEATURE_RANKING_ALGORITHM}-${TIME_LIMIT}t-${EVAL_METRIC} get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/feature_ranking.py --directory ${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME}\
	                                                     --algorithm ${FEATURE_RANKING_ALGORITHM}\
														 --cpus ${CORES}\
														 --time-limit ${TIME_LIMIT}\
														 --metric ${EVAL_METRIC}\
														 --mode ${RANKING_RECURSIVENESS}\
														 $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-feature-subset-stochastic-search:
	@TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${FEATURE_RANKING_ALGORITHM}-${SUBSET_LEFT_TAKEN_POLICIY}-${SUBSETS_SIZE_AMOUNT}s-${EVAL_METRIC} get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/feature_subset_stochastic_search.py --directory ${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_${FEATURE_RANKING_ALGORITHM}\
	                                                                      --subset-size ${SUBSETS_SIZE_AMOUNT}\
																		  --left-policy ${SUBSET_LEFT_TAKEN_POLICIY}\
																		  --attempts ${SUBSETS_TRIALS}\
																		  $$(for x in ${BEST_FEATURES_FOR_SUBSET_AMOUNTS};do echo -n "--best $${x} ";done)\
																		  --cpus ${CORES}\
																		  --metric ${EVAL_METRIC}\
																		  $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-model-pruner-search:
	@TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${MODEL_PRUNING_ALGORITHM}-${PERFORMANCE_DEGRADATION}d-${EVAL_METRIC}-search get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/model_pruner_search.py search --directory ${DATASETS_DIRNAME}/$${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME}\
	                                                                --algorithm ${MODEL_PRUNING_ALGORITHM}\
	                                                                --attempts ${MODEL_PRUNING_TRIALS}\
																	$$(for x in ${MODEL_PRUNING_AMOUNTS};do echo -n "--amount $${x} ";done)\
																	--cpus ${CORES}\
																	--subsets $${SUBSETS_TARGET_DIR}\
																	--subsets-degradation ${PERFORMANCE_DEGRADATION}\
																	--metric ${EVAL_METRIC}\
																	--top-subsets ${MODELS_PRUNED_FOR_SUBSETS_MAX}\
																	$${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-model-pruner-test:
	@TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${FEATURE_RANKING_ALGORITHM}-${SUBSET_LEFT_TAKEN_POLICIY}-${SUBSETS_SIZE_AMOUNT}s-${MODEL_PRUNING_ALGORITHM}-${PERFORMANCE_DEGRADATION}d-${EVAL_METRIC}-test get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/model_pruner_search.py test --directory ${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}_${PROBLEM_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_${ALGORITHM}/feature_subset_stochastic_search_${SUBSET_LEFT_TAKEN_POLICIY}/feature_subsets_${SUBSETS_SIZE_AMOUNT}s\
	                                                              --algorithm ${MODEL_PRUNING_ALGORITHM}\
																  --cpus ${CORES}\
																  --subsets-degradation ${PERFORMANCE_DEGRADATION}\
																  --models-degradation ${PERFORMANCE_DEGRADATION}\
																  --metric ${EVAL_METRIC}\
																  --top-subsets ${MODELS_PRUNED_FOR_SUBSETS_MAX}\
																  $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-plot-feature-ranking:
    @TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${PLOT_METRIC}-plot get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/plotter.py plot_rfe --directory ${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_${FEATURE_RANKING_ALGORITHM}\
	                                                      --metric ${PLOT_METRIC}\
														  $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-plot-feature-subset-stochastic-search:
    @TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${PLOT_METRIC}-plot get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/plotter plot_ss --directory ${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_${FEATURE_RANKING_ALGORITHM}/feature_subset_stochastic_search_${SUBSET_LEFT_TAKEN_POLICIY}\
	                                                  --metric ${PLOT_METRIC}\
													  $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-plot-model-pruner-search:
    @TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${PLOT_METRIC}-plot-search get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/plotter.py plot_mp --directory ${DATASETS_DIRNAME}/${DATASET}/${COMPOSITION}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/prune_search/\
	                                                     --subsets-dir ${SUBSETS_DIR}\
														 --metric ${PLOT_METRIC}\
														 $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-plot-model-pruner-test:
	@TMP=$$(${MAKE} DATASET=${DATASET} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${MODEL_PRUNING_ALGORITHM}-${PLOT_METRIC}-plot-test get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/plotter.py plot_mt --directory ${DIR_OF_INTEREST}\
	                                                     --algorithm ${MODEL_PRUNING_ALGORITHM}\
														 --metric ${PLOT_METRIC}\
														 --degradation ${PERFORMANCE_DEGRADATION}\
														 $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

#######################################################
##### RUN EITHER LOCAL OR FROM JBSUB-ENABLED ENV ######
#######################################################

run-all-dataset-creator:
	for f in ${LIST_DATASET}; do\
		${MAKE} DATASET=$${f} run-dataset-creator;\
	done

run-all-automl-search:
	for f in ${LIST_DATASET}; do\
		${MAKE} DATASET=$${f} run-automl-search;\
	done

run-all-automl-dump:
	for f in ${LIST_DATASET}; do\
			${MAKE} DATASET=$${f} run-automl-dump;\
	done

run-all-feature-ranking:
	for f in ${LIST_DATASET}; do\
		${MAKE} DATASET=$${f} run-feature-ranking;\
	done

run-all-feature-subset-stochastic-search:
	for f in ${LIST_DATASET}; do\
		for s in ${LIST_SUBSETS_SIZE_AMOUNTS}; do\
			${MAKE} DATASET=$${f} SUBSETS_SIZE_AMOUNT=$${s} run-feature-subset-stochastic-search;\
		done;\
	done

run-all-model-pruner-search:
	for f in ${LIST_DATASET}; do\
		for malg in ${LIST_MODEL_PRUNING_ALGORITHMS}; do\
			if [[ $${malg} == *"_for_subset"* ]]; then\
				for s in ${LIST_SUBSETS_SIZE_AMOUNTS}; do\
					${MAKE} DATASET=$${f} MODEL_PRUNING_ALGORITHM=$${malg} SUBSETS_TARGET_DIR=${DATASETS_DIRNAME}/$${f}/$${ct}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_${FEATURE_RANKING_ALGORITHM}/feature_subset_stochastic_search_$${SUBSET_LEFT_TAKEN_POLICIY}/feature_subsets_$${s}s run-model-pruner-search;\
				done;\
			else\
				${MAKE} DATASET=$${f} MODEL_PRUNING_ALGORITHM=$${malg} SUBSETS_TARGET_DIR="" run-model-pruner-search;\
			fi;\
		done;\
	done

run-all-model-pruner-test:
	for f in ${LIST_DATASET}; do\
		for alg in ${LIST_MODEL_PRUNING_ALGORITHMS}; do\
			for s in ${LIST_SUBSETS_SIZE_AMOUNTS}; do\
				${MAKE} DATASET=$${f} SUBSETS_SIZE_AMOUNT=$${s} MODEL_PRUNING_ALGORITHM=$${alg} run-model-pruner-test;\
			done;\
		done;\
	done


run-all-print:
	for f in ${LIST_DATASET}; do\
		${MAKE} DATASET=$${f} run-plot-feature-ranking;\
		${MAKE} DATASET=$${f} run-plot-feature-subset-stochastic-search;\
		for alg in ${LIST_MODEL_PRUNING_ALGORITHMS}; do\
			echo "";\
		done;\
		${MAKE} DATASET=$${f} SUBSETS_DIR=$${SUBSETS} run-plot-model-pruner-search;\
	done


check-jobs:
    ifeq (${IS_MISS_JBSUB}, 1)
	@echo "Currently" $$(( `ps aux | grep ${USER} | grep -e "automl_model_search" -e "dataset_creator" -e "feature_ranking" -e "feature_subset_stochastic_search" -e "model_pruner_search" | wc -l` - 3 )) "jobs"
    else
	@echo "Currently" $$(( `jbinfo -state run | grep ${USER} | wc -l` + `jbinfo -state pend | grep ${USER} | wc -l` )) "jobs"
    endif


stop-jobs:
    ifeq (${IS_MISS_JBSUB}, 1)
	@pkill -f automl_model_search.py || true;\
	pkill -f dataset_creator.py || true;\
	pkill -f feature_ranking.py || true;\
	pkill -f feature_subset_stochastic_search.py || true;\
	pkill -f model_pruner_search.py || true
    else
	@for jid in `jbinfo -state pend | awk 'FNR > 2 { print $$1 }'`; do\
		jbadmin -kill $${jid};\
	done;\
	for jid in `jbinfo -state run | awk 'FNR > 2 { print $$1 }'`; do\
		jbadmin -kill $${jid};\
	done
    endif
