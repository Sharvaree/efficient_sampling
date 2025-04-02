#!/bin/sh
# conda activate edm
# python eval_conditional_qm9.py --uncond_generators_path outputs/edm_qm9 --generators_path outputs/exp_cond_mu --classifiers_path qm9/property_prediction/outputs/exp_class_mu --property mu  --iterations 4  --batch_size 1 --task optimize_prior


python eval_analyze.py -m +mode=submitit_multi_node
    