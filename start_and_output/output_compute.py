import tensorflow as tf


def compute_loss(y_true: tf.Tensor, loss: tf.Tensor) -> tf.Tensor:
	return loss


class ComputeMetrics(tf.keras.metrics.Metric):
	def __init__(self, save_path: str, return_metrics: str, thresholds) -> None:
		super().__init__(name=return_metrics)
		self.auc_layer = tf.keras.metrics.AUC(curve='ROC', name='auc')
		self.precision_layer = tf.keras.metrics.Precision(name='precision', thresholds=thresholds)
		self.recall_layer = tf.keras.metrics.Recall(name='recall', thresholds=thresholds)
		self.accuracy_layer = tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=thresholds)
		self.save_path: str = save_path
		self.results: dict = dict()
		self.return_metrics: str = return_metrics
	
	def update_state(
			self,
			y_true: tf.Tensor,
			y_probs: tf.Tensor,
			sample_weight: bool = None) -> None:
		self.auc_layer.update_state(y_true, y_probs)
		self.precision_layer.update_state(y_true, y_probs)
		self.recall_layer.update_state(y_true, y_probs)
		self.accuracy_layer.update_state(y_true, y_probs)
		self.results['auc'] = self.auc_layer.result()
		self.results['precision'] = self.precision_layer.result()
		self.results['recall'] = self.recall_layer.result()
		self.results['accuracy'] = self.accuracy_layer.result()
		self.write_metrics()
	
	def write_metrics(self) -> None:
		results_sort: list = ['auc', 'precision', 'recall', 'accuracy']
		written_line: str = '\t'.join(map(str, [self.results[result_type].numpy() for result_type in results_sort]))
		print(f"\n{written_line=}")
		with open(self.save_path, 'a') as f:
			f.write(written_line)
	
	def result(self) -> tf.Tensor:
		return self.results[self.return_metrics]
	
	def reset_state(self):
		# 重置所有子指标的状态
		self.accuracy_layer.reset_state()
		self.precision_layer.reset_state()
		self.recall_layer.reset_state()
		self.auc_layer.reset_state()
		