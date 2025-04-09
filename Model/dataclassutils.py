from dataclasses import dataclass
import tensorflow as tf
from transformers import TFT5Model



@dataclass
class MaskedInputs:
	protein_name: tf.string
	input_ids: tf.Tensor
	input_mask: tf.Tensor
	masked_lm_positions: tf.Tensor or None
	masked_lm_ids: tf.Tensor or None
	masked_lm_weights: tf.Tensor or None
	labels: tf.Tensor


@dataclass
class MLMOutput:
	logits: tf.Tensor
	probs: tf.Tensor
	loss: tf.Tensor
	per_example_loss: tf.Tensor
	preds: tf.Tensor