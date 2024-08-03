SHELL:=/bin/bash
VENV:=../../.venv
PYTHON:=$(VENV)/bin/python3
MAKEFLAGS+=--no-print-directory

LIMITED_TRAFFIC="--limited-traffic"
CONFIG?=./config.json
SEED?=42
PARALLEL?=1
MAX_PROG?=1

pipeline:
	SUBSETS_RATIOS=$$(cat ${CONFIG} | ${PYTHON} -c "import sys, json; print(' '.join(str(x) for x in json.load(sys.stdin)['target_subset_ratios']))");\
	SCENARIOS=$$(cat ${CONFIG} | ${PYTHON} -c "import sys, json; print(' '.join(str(x) for x in json.load(sys.stdin)['scenarios']))");\
	PERF_DROPS=$$(cat ${CONFIG} | ${PYTHON} -c "import sys, json; print(' '.join(str(x) for x in json.load(sys.stdin)['performance_drops']))");\
	CMD="${PYTHON} main.py -c ${CONFIG} -s ${SEED} -p ${PARALLEL} -a ";\
	$${CMD} train &&\
		( $${CMD} baselines &\
		( $${CMD} compare_pruning &\
		$${CMD} recursive_subset_search --fixed-rank --zero-first ${LIMITED_TRAFFIC} &\
		$${CMD} recursive_subset_search --fixed-rank ${LIMITED_TRAFFIC} &\
		$${CMD} recursive_subset_search --zero-first ${LIMITED_TRAFFIC} &\
		$${CMD} recursive_subset_search ${LIMITED_TRAFFIC} &\
		( $${CMD} run_scenario --scenario-name o_to_o ${LIMITED_TRAFFIC} ) &\
		( $${CMD} only_pruning ${LIMITED_TRAFFIC} && ( for d in $${PERF_DROPS}; do ($${CMD} run_scenario --scenario-name o_to_po ${LIMITED_TRAFFIC} --performance-drop $${d} &); done ) ) &\
		( for r in $${SUBSETS_RATIOS}; do ($${CMD} stochastic_search_then_pruning --subset-ratio $${r} ${LIMITED_TRAFFIC} && ( for d in $${PERF_DROPS}; do ($${CMD} run_scenario --scenario-name o_to_ec ${LIMITED_TRAFFIC} --subset-ratio $${r} --performance-drop $${d} &); done ) &); done ) &\
		( for r in $${SUBSETS_RATIOS}; do ($${CMD} only_stochastic_search --subset-ratio $${r} ${LIMITED_TRAFFIC} && ( for d in $${PERF_DROPS}; do ($${CMD} run_scenario --scenario-name o_to_eo ${LIMITED_TRAFFIC}  --subset-ratio $${r} --performance-drop $${d} &); done ) &); done ) &\
		( for r in $${SUBSETS_RATIOS}; do ($${CMD} pruning_then_stochastic_search --subset-ratio $${r} ${LIMITED_TRAFFIC} && ( for d in $${PERF_DROPS}; do ($${CMD} run_scenario --scenario-name o_to_ec ${LIMITED_TRAFFIC} --subset-ratio $${r} --performance-drop $${d} --prune-first &); done ) &); done ) & )

dataset:
	${PYTHON} main.py -c ${CONFIG} -s ${SEED} -p ${PARALLEL} -v -a dataset

correlations:
	${PYTHON} main.py -c ${CONFIG} -s ${SEED} -p ${PARALLEL} -v -a correlations

evaluation:
	PRUNING_AMOUNTS=$$(cat ${CONFIG} | ${PYTHON} -c "import sys, json; print(' '.join(str(x) for x in json.load(sys.stdin)['prune_ratios']))");\
	CMD="${PYTHON} main.py -c ${CONFIG} -s ${SEED} -p ${PARALLEL} -v --limited-traffic -a ";\
	$${CMD} train_evaluation &&\
	$${CMD} test_evaluation &&\
	( for r in $${PRUNING_AMOUNTS}; do ($${CMD} train_evaluation -pr $${r} && $${CMD} test_evaluation -pr $${r}); done )

ps:
	ps aux | grep ${PYTHON}

cnt:
	@wcExp=$(${MAKE} ps | wc -l);\
	((wcExp=`${MAKE} ps | wc -l`-3));\
	echo "Number of processing running: $${wcExp}"

kill:
	pkill -f ${PYTHON}

all:
	SEEDS=$$(cat ${CONFIG} | ${PYTHON} -c "import sys, json; print(' '.join(str(x) for x in json.load(sys.stdin)['seeds']))");\
	COUNT=0;\
	for i in $${SEEDS}; do\
		${MAKE} SEED=$${i} pipeline &\
		FOO_PID=$$!;\
		let "COUNT+=1";\
		if test $${COUNT} -eq ${MAX_PROG}; then\
			wait $${FOO_PID};\
			let "COUNT-=MAX_PROG+1";\
		fi;\
	done;\
	wait
