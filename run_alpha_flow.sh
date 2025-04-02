#!/bin/sh
# conda activate edm
# python eval_conditional_qm9.py --uncond_generators_path outputs/edm_qm9 --generators_path outputs/exp_35_conditional_nf192_9l_alpha --classifiers_path qm9/property_prediction/outputs/exp_class_alpha --property alpha  --iterations 4  --batch_size 1 --task optimize_prior


python eval_conditional_qm9.py -m \
    +mode=submitit_multi_node \
    uncond_generators_path=outputs/ecnf_qm9_new \
    model_type=velocity \
    generators_path=outputs/exp_35_conditional_nf192_9l_alpha \
    classifiers_path=qm9/property_prediction/outputs/exp_class_alpha \
    property=alpha  \
    iterations=99  \
    batch_size=1 \
    task=optimize_prior \
    run_idx='range(0,50)' \
    
