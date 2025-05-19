from dataclasses import dataclass
import tensorflow as tf


@dataclass
class MaskInputs:
	protein_name: tf.string
	input_ids: tf.Tensor
	input_mask: tf.Tensor
	labels: tf.Tensor
	
	
@dataclass
class MaskedInputs:
	protein_name: tf.string
	input_ids: tf.Tensor
	input_mask: tf.Tensor
	labels: tf.Tensor
	masked_lm_positions: tf.Tensor or None
	masked_lm_ids: tf.Tensor or None
	masked_lm_weights: tf.Tensor or None


@dataclass
class Output:
	logits: tf.Tensor
	probs: tf.Tensor
	loss: tf.Tensor
	preds: tf.Tensor
	