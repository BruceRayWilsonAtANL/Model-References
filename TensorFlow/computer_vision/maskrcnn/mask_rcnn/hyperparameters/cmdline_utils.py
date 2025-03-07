#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defining common model params used across all the models."""

from absl import flags


def define_hparams_flags():

    flags.DEFINE_string(
        'log_path',
        default="./mrcnn.json",
        help=(
            'The path where dllogger json file will be saved. Please include the'
            ' name of the json file as well.'
        )
    )

    flags.DEFINE_string('checkpoint',
        default='weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603',
        help='Checkpoint filepath'
    )

    flags.DEFINE_integer(
        'eval_batch_size',
        default=4,
        help='Batch size for evaluation.'
    )

    flags.DEFINE_integer(
        'skip_cycles',
        default=0,
        help='Number of EVAL cycles to skip initially while resuming from checkpoint'
             ' in train_and_eval mode. (1 cycle = --num_steps_per_eval steps).'
    )

    flags.DEFINE_bool(
        'eval_after_training',
        default=True,
        help='Run one eval after the training finishes.'
    )

    flags.DEFINE_integer('eval_samples', default=5000, help='Number of training steps')

    flags.DEFINE_bool(
        'include_groundtruth_in_features',
        default=False,
        help=(
            'If `val_json_file` is not provided, one can also read groundtruth'
            ' from input by setting `include_groundtruth_in_features`=True'
        )
    )

    # Gradient clipping is a fairly coarse heuristic to stabilize training.
    # This model clips the gradient by its L2 norm globally (i.e., across
    # all variables), using a threshold obtained from multiplying this
    # parameter with sqrt(number_of_weights), to have a meaningful value
    # across both training phases and different sizes of imported modules.
    # Refer value: 0.02, for 25M weights, yields clip norm 10.
    # Zero or negative number means no clipping.
    flags.DEFINE_float("global_gradient_clip_ratio", default=-1.0, help="Global Gradient Clipping Ratio")

    flags.DEFINE_float("init_learning_rate", default=5e-3, help="Initial Learning Rate")

    flags.DEFINE_float("warmup_learning_rate", default=0., help="Warmup Learning Rate Decay Factor")

    flags.DEFINE_bool('finetune_bn', False, 'is batchnorm training mode')

    flags.DEFINE_float("l2_weight_decay", default=1e-4, help="l2 regularization weight")

    flags.DEFINE_string('mode', default='train_and_eval', help='Mode to run: train or eval')

    flags.DEFINE_string('pyramid_roi_impl', default='habana', help='Implementation to use for PyramidRoiAlign: [habana | habana_fp32 | gather]')

    flags.DEFINE_string(
        'model_dir',
        default='results/',
        help='The directory where the model and training/evaluation summaries are stored.'
    )

    flags.DEFINE_float("momentum", default=0.9, help="Optimizer Momentum")

    flags.DEFINE_integer('num_steps_per_eval', default=29568, help='Number of steps per evaluation epoch.')

    flags.DEFINE_integer('save_checkpoints_steps', default=2500, help='Save a checkpoint every N steps.')

    flags.DEFINE_integer('seed', default=None, help='Set a debug seed for reproducibility.')

    flags.DEFINE_bool('deterministic', False, help='Enable deterministic behavior. Needs --seed to be set.')

    flags.DEFINE_integer('train_batch_size', default=4, help='Batch size for training.')

    flags.DEFINE_integer(
        'total_steps',
        default=360000,
        help=(
            'The number of steps to use for training. This flag'
            ' should be adjusted according to the --train_batch_size flag.'
        )
    )

    flags.DEFINE_list(
        'learning_rate_decay_levels',
        default=['0.1', '0.01'],
        help=(
            'The learning rate decay levels which modify the learning rate using the formula:'
            ' `lr = decay * init_lr`. Decay factor applied at learning_rate_steps.'
        )
    )
    flags.DEFINE_list(
        'learning_rate_steps',
        default=['240000', '320000'],
        help=(
            'The steps at which learning rate changes. This flag'
            ' should be adjusted according to the --train_batch_size flag.'
        )
    )
    flags.DEFINE_integer('warmup_steps', default=1000, help='The number of steps to use warmup learning rate for')

    flags.DEFINE_bool('amp', default=False, help='Enable automatic mixed precision')

    flags.DEFINE_bool(
        'use_batched_nms',
        default=False,
        help='Enable Batched NMS.'
    )

    flags.DEFINE_bool(
        'nms_on_hpu',
        default=True,
        help='Enable NMS on HPU'
    )

    flags.DEFINE_bool(
        'topk_before_nms',
        default=True,
        help='Run topk before NMS'
    )

    flags.DEFINE_bool(
        'offload_maskhead_postproc',
        default=False,
        help='WA to offload ScatterNd in MaskHead Post Processing to CPU'
    )

    flags.DEFINE_bool(
        'use_habana_get_mask_targets',
        default=True,
        help='Use HPU-optimized version of \'get_mask_targets\''
    )

    flags.DEFINE_bool(
        'use_custom_box_proposals_op',
        default=False,
        help='Use GenerateBoundingBoxProposals op.'
    )

    flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

    flags.DEFINE_bool(
        'use_tf_distributed',
        default=False,
        help='Use tensorflow distributed API'
    )

    flags.DEFINE_bool('xla', default=False, help='Enable XLA JIT Compiler.')

    flags.DEFINE_string('training_file_pattern',
        default="/data/tensorflow/coco2017/tf_records/train-*.tfrecord",
        help='TFRecords file pattern for the training files'
    )

    flags.DEFINE_string('validation_file_pattern',
        default="/data/tensorflow/coco2017/tf_records/val-*.tfrecord",
        help='TFRecords file pattern for the validation files'
    )

    flags.DEFINE_string('val_json_file',
        default="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json",
        help='Filepath for the validation json file'
    )

    flags.DEFINE_string('device', default="HPU", help='Select device from: "HPU", "CPU", "GPU"')

    flags.DEFINE_integer('save_summary_steps', default=None, help='Steps between saving summaries to TensorBoard.')

    flags.DEFINE_string('profile', default="", help='Gather TensorBoard profiling data. Must be an integer or a pair of comma-separated integers. For example: --profile 4,6')

    flags.DEFINE_string('recipe_cache',
        default='/tmp/maskrcnn_recipe_cache/',
        help='Path to recipe cache directory. Set to empty to disable recipe cache. Externally set \'TF_RECIPE_CACHE_PATH\' will override this setting.'
    )

    flags.DEFINE_integer('max_cpus', default=16, help='Max number of CPUs used by process.')

    ############################# TO BE REMOVED ###################################

    flags.DEFINE_integer(
        'report_frequency',
        default=None,
        help='The amount of batches in between accuracy reports at evaluation time'
    )

    ############################# TO BE REMOVED ###################################

    ############################### ISSUES TO FIX - FLAGS #############################"

    # TODO: Remove when XLA at inference fixed
    flags.DEFINE_bool(
        'allow_xla_at_inference',
        default=False,
        help='Enable XLA JIT Compiler at Inference'
    )

    return flags.FLAGS
