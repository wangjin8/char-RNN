import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN
import os
# from IPython import embed

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'model1/default/converter.pkl', 'model/default/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model1/default/model-1000', 'model/default/')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 3000, 'max length to generate')

# FLAGS.converter_path='model1/default/converter.pkl'
# FLAGS.checkpoint_path='model1/default/'
# FLAGS.max_length=1000

def main(_):
    FLAGS.start_string = FLAGS.start_string.encode('utf-8').decode('utf-8')
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr_to_text(arr))


if __name__ == '__main__':
    tf.app.run()
