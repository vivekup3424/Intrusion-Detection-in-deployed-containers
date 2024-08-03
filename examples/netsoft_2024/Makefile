SHELL:=/bin/bash
VENV:=../../.venv
PYTHON:=$(VENV)/bin/python3
MAKEFLAGS+=--no-print-directory

CONFIG?=./config_cicids2017.json
SEED?=42
PARALLEL?=1
MAX_PROG?=1

pipeline:
	PRUNING_AMOUNTS=$$(cat ${CONFIG} | ${PYTHON} -c "import sys, json; print(' '.join(str(x) for x in json.load(sys.stdin)['pruning_amounts']))");\
	CMD="${PYTHON} main.py -c ${CONFIG} -s ${SEED} -p ${PARALLEL} -a ";\
	( $${CMD} train_ol &&\
		( $${CMD} test_ol &\
		( for r in $${PRUNING_AMOUNTS}; do ($${CMD} prune_ol -r $${r} &); done ) &\
		( for r in $${PRUNING_AMOUNTS}; do ( ( $${CMD} train_ol_pruned -r $${r} && $${CMD} test_ol_pruned -r $${r} ) & ); done ) & ) ) &\
	$${CMD} train_tl &&\
		( $${CMD} test_tl &\
		( for r in $${PRUNING_AMOUNTS}; do ($${CMD} prune_tl -r $${r} &); done ) &\
		for r in $${PRUNING_AMOUNTS}; do $${CMD} train_tl_pruned -r $${r} && $${CMD} test_tl_pruned -r $${r}; done )

evaluation:
	PRUNING_AMOUNTS=$$(cat ${CONFIG} | ${PYTHON} -c "import sys, json; print(' '.join(str(x) for x in json.load(sys.stdin)['pruning_amounts']))");\
	CMD="${PYTHON} main.py -c ${CONFIG} -s ${SEED} -p ${PARALLEL} -v -a ";\
	$${CMD} evaluate_train &&\
	$${CMD} evaluate_test &&\
	( for r in $${PRUNING_AMOUNTS}; do ($${CMD} evaluate_train -r $${r} && $${CMD} evaluate_test -r $${r}); done )

datadrift:
	${PYTHON} main.py -c ${CONFIG} -s ${SEED} -p ${PARALLEL} -v -a datadrift

feature-importance:
	${PYTHON} main.py -c ${CONFIG} -s ${SEED} -p ${PARALLEL} -v -a feature_importance
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
