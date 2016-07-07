import tensorflow as tf
import time
import re
import os
from datetime import datetime
from itertools import islice



# To view intermediate values how to? - dump the values as histogram or values
# How to control the number of inputs to be used for 

FILE_PATH = "../data/fb/distributed_record_local_label/drll0.csv"
FID_LOID_2_LAID = "../data/fb/fileid_localid_to_labelid"


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('train_dir', './train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of entries in one batch""")

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def read_record(filename_list):
	filename_queue = tf.train.string_input_producer(filename_list)
	num_preprocess_threads = 16
	min_queue_examples = FLAGS.batch_size
	class FbRecord(object):
		pass;

	result = FbRecord()

	reader = tf.TextLineReader(skip_header_lines=1)
	#key value is the serial number and the value is the other five attributes
	key, value = reader.read(filename_queue)
	#print key.dtype, value.dtype, value.value_index, value.graph, value.op, value.consumers()
	# The type conversion from string to float depends on the default value
	record = tf.decode_csv(value,[[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
	#print "record",len(record)
	#tf.shape(record), tf.size(record), tf.rank(record)

	sno, x, y, a, t, p = record
	feature = tf.pack([x, y, a, t])
	label = p
	#print result.feature.dtype, result.label.dtype
	#print tf.shape(feature), tf.size(feature), tf.rank(feature)
	#print tf.shape(label), tf.size(label), tf.rank(label)
	
	features, label_batch = tf.train.batch(
        [feature, label],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size)
	#print tf.shape(features), tf.size(features), tf.rank(features)

	return features, label_batch

def get_num_classes(filename, fileid):
	with open(filename) as infile:
		lineid = 0
		for line in islice(infile,1,None):
			if(lineid == fileid):
				classes = line.strip().split(",")
				#print classes
				return len(classes)
			lineid = lineid + 1

def get_num_records(filename):
	with open(filename) as f:
		for i, l in enumerate(f):
			pass
	return i

NUM_RECORDS = get_num_records(FILE_PATH)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUM_RECORDS

def inference(features):
  NUM_FEATURES = 4
  NUM_HIDDEN_NODES = 100
  NUM_CLASSES = get_num_classes(FID_LOID_2_LAID, 0)
  print NUM_RECORDS, NUM_CLASSES

	# local
  with tf.variable_scope('local') as scope:
    weights = _variable_with_weight_decay('weights', shape=[NUM_FEATURES, NUM_HIDDEN_NODES],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [NUM_HIDDEN_NODES], tf.constant_initializer(0.1))
    local = tf.nn.relu(tf.matmul(features, weights) + biases, name=scope.name)
    _activation_summary(local)
   # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [NUM_HIDDEN_NODES, NUM_CLASSES],
                                          stddev=1.0/NUM_HIDDEN_NODES, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def lossfn(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  
  prediction = tf.nn.softmax(logits);
  in_top_k = tf.nn.in_top_k(prediction, labels, 3, name='is_target')
  convert_to_float = tf.to_float(in_top_k)
  label_mean = tf.reduce_mean(convert_to_float)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return (tf.add_n(tf.get_collection('losses'), name='total_loss'), label_mean)


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def main(argv=None):
	#
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable=False)
		features, label_batch = read_record([FILE_PATH])
		logits = inference(features)
		loss, target_mean = lossfn(logits, label_batch)
		train_op = train(loss, global_step)
		# Create a saver.
		saver = tf.train.Saver(tf.all_variables())
		# Build the summary operation based on the TF collection of Summaries.
		summary_op = tf.merge_all_summaries()

		# Build an initialization operation to run below.
		init = tf.initialize_all_variables()
		# Start running operations on the Graph.
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
		sess.run(init)

		# Start the queue runners.
		tf.train.start_queue_runners(sess=sess)

		summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

		start_time = time.time()
		for step in xrange(FLAGS.max_steps):
			# features_list, labels_list = sess.run([features, label_batch])
			_,loss_value, target_mean_value = sess.run([train_op, loss, target_mean])
			if step%100 == 0:
				duration = time.time() - start_time
				print "time take to iterate ", step, duration
				start_time = time.time()
				num_examples_per_step = FLAGS.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)

				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
				              'sec/batch), %f ')
				print (format_str % (datetime.now(), step, loss_value,
				                     examples_per_sec, sec_per_batch, target_mean_value))
			#print len(features_list), len(labels_list)
			#print (features_list.shape), (labels_list.shape)

			#assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
			if step % 100 == 0:
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)

			# Save the model checkpoint periodically.
			if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)




if __name__ == '__main__':
	tf.app.run()



























