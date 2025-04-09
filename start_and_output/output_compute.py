import tensorflow as tf


def compute_metrics(y_true, y_pred, save_path):
	auc_layer = tf.keras.metrics.AUC(curve='ROC', name='auc')
	acc_layer = tf.keras.metrics.Accuracy(name='acc')
	precision_layer = tf.keras.metrics.Precision(name='precision')
	recall_layer = tf.keras.metrics.Recall(name='recall')
	accuracy_layer = tf.keras.metrics.Accuracy(name='accuracy')
	auc_layer.update_state(y_true, y_pred)
	acc_layer.update_state(y_true, y_pred)
	precision_layer.update_state(y_true, y_pred)
	recall_layer.update_state(y_true, y_pred)
	accuracy_layer.update_state(y_true, y_pred)
	auc = auc_layer.result().numpy()
	acc = acc_layer.result().numpy()
	precision = precision_layer.result().numpy()
	recall = recall_layer.result().numpy()
	accuracy = accuracy_layer.result().numpy()
	print(f'auc: {auc}, acc: {acc}, precision: {precision}, recall: {recall}, accuracy: {accuracy}')
	with open(save_path, 'a') as f:
		f.write(f'{auc}\t{acc}\t{precision}\t{recall}\t{accuracy}\n')
