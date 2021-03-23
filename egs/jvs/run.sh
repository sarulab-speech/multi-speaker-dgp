#!/bin/bash

stage=0
stop_stage=4

prjdir=../..
db_root=jvs_ver1
datadir=data
dumpdir=dump
dump_raw_dir=${dumpdir}/raw
dump_norm_dir=${dumpdir}/norm
conf=conf/dgp.yaml
resume=""
duration_checkpoint=""
acoustic_checkpoint=""

subsets=(train_nodev dev eval)

. ${prjdir}/utils/parse_options.sh || exit 1;

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Make speaker list & perform train/dev/eval split"
    mkdir -p ${datadir}
    . local/make_speaker_list.sh > ${datadir}/speakers.txt
    python -uB local/make_subset.py \
        --db-root ${db_root} \
        --outdir ${datadir} \
        --config ${conf}
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Preprocessing"
    for s in ${subsets[@]}; do
        python -uB ${prjdir}/preprocess.py \
            --lab-scp ${datadir}/lab_${s}.scp \
            --wav-scp ${datadir}/wav_${s}.scp \
            --question-path ${prjdir}/utils/questions_jp.hed \
            --dumpdir ${dump_raw_dir}/${s} \
            --config ${conf}
    done

    for xy in X Y; do
        for typ in duration acoustic; do
            if [ ${xy} = 'X' ]; then
                stats_typ=minmax
            else
                stats_typ=meanvar
            fi
            echo "Compute statistics for ${xy}_${typ}. stats_typ: ${stats_typ}"
            python -uB ${prjdir}/compute_statistics.py \
                --in-dir ${dump_raw_dir}/train_nodev/${xy}_${typ} \
                --out-path ${dumpdir}/${xy}_${typ}.pkl \
                --type ${stats_typ}
            echo "Perform normalization for ${xy}_${typ}."
            for s in ${subsets[@]}; do
                python -uB ${prjdir}/normalize.py \
                    --in-dir ${dump_raw_dir}/${s}/${xy}_${typ} \
                    --scaler ${dumpdir}/${xy}_${typ}.pkl \
                    --dumpdir ${dump_norm_dir}/${s}/${xy}_${typ}
            done
        done
    done

fi

expdir=exp/$(basename ${conf} .yaml)
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Train phoneme duration model"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp ${dumpdir}/X_duration.pkl ${expdir}
    cp ${dumpdir}/Y_duration.pkl ${expdir}

    python -uB ${prjdir}/train.py \
        --train-in-dir ${dump_norm_dir}/train_nodev/X_duration \
        --train-out-dir ${dump_norm_dir}/train_nodev/Y_duration \
        --dev-in-dir ${dump_norm_dir}/dev/X_duration \
        --dev-out-dir ${dump_norm_dir}/dev/Y_duration \
        --outdir ${expdir} \
        --config ${conf} \
        --speakers ${datadir}/speakers.txt \
        --mode duration \
        --resume ${resume} \
        >& ${expdir}/train_duration.log
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Train acoustic model"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp ${dumpdir}/X_acoustic.pkl ${expdir}
    cp ${dumpdir}/Y_acoustic.pkl ${expdir}

    python -uB ${prjdir}/train.py \
        --train-in-dir ${dump_norm_dir}/train_nodev/X_acoustic \
        --train-out-dir ${dump_norm_dir}/train_nodev/Y_acoustic \
        --dev-in-dir ${dump_norm_dir}/dev/X_acoustic \
        --dev-out-dir ${dump_norm_dir}/dev/Y_acoustic \
        --outdir ${expdir} \
        --config ${conf} \
        --speakers ${datadir}/speakers.txt \
        --mode acoustic \
        --resume ${resume} \
        >& ${expdir}/train_acoustic.log
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Decoding"
    [ -z ${duration_checkpoint} ] && duration_checkpoint="$(ls -dt "${expdir}"/checkpoint-duration-*.pkl | head -1)"
    [ -z ${acoustic_checkpoint} ] && acoustic_checkpoint="$(ls -dt "${expdir}"/checkpoint-acoustic-*.pkl | head -1)"
    durname=$( basename ${duration_checkpoint} .pkl | sed 's/checkpoint-//g' )
    aconame=$( basename ${acoustic_checkpoint} .pkl | sed 's/checkpoint-//g' )
    outdir=${expdir}/results/${durname}_${aconame}
    [ ! -e ${outdir} ] && mkdir -p ${outdir}

    python -uB ${prjdir}/synthesize.py \
        --labdir ${datadir}/lab_eval.scp \
        --outdir ${outdir} \
        --duration-checkpoint ${duration_checkpoint} \
        --acoustic-checkpoint ${acoustic_checkpoint} \
        --speakers ${datadir}/speakers.txt \
        --question-path ${prjdir}/utils/questions_jp.hed \
        >& ${outdir}/synthesis.log
    echo "Successfully finished speech synthesis"
fi
