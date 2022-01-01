import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model 模型名')
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch ')#一个batch里面的序列数量
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq ')#序列的长度
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm LSTM隐层的大小')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers LSTM的层数')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding 是否使用 embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding embedding的大小')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate  学习率')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training  训练期间的dropout比率')
tf.flags.DEFINE_string('input_file', '', 'utf8 encoded text file  utf8编码过的text文件')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')   #一个step 是运行一个batch， max_steps固定了最大的运行步数
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')  #每隔1000步会将模型保存下来
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')  #每隔10步会在屏幕上打出曰志
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')#最大字符数量
# 使用的字母（汉字）的最大个数。默认为3500 。程序会自动挑选出使用最多的字，井将剩下的字归为一类，并标记为＜unk＞

# FLAGS.input_file = 'data/shakespeare.txt'
FLAGS.input_file ="data/poetry.txt"
FLAGS.max_steps = 1000

def main(_):
    model_path = os.path.join('model1', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    print(FLAGS.input_file)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, FLAGS.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    print(converter.vocab_size)
    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()
