import os
import argparse
import datetime
import tensorflow as tf
from tensorflow.contrib import slim

import config as cfg
from yolonet_v1 import YOLONet
from timer import Timer
from pascal_voc import pascal_voc



class Solver(object):
    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER  # 最大迭代次数

        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE

        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        # 创建文件夹保存数据
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(var_list=self.variable_to_restore, max_to_keep=None)
        # 模型参数保存路径
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')

        # tensorboard
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        # global_step
        self.global_step = tf.train.create_global_step()
        # 采用学习率衰减
        self.learning_rate = tf.train.exponential_decay(learning_rate=self.initial_learning_rate,
                                                        global_step=self.global_step, decay_steps=self.decay_steps,
                                                        decay_rate=self.decay_rate, staircase=self.staircase,
                                                        name='learning_rate')
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='Adam')
        self.train_op = slim.learning.create_train_op(total_loss=self.net.total_loss, optimizer=self.optimizer,
                                                      global_step=self.global_step)
        # 等价于
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.net.total_loss,
        #                                                                                               global_step=self.global_step)

        # GPU加速
        config = tf.ConfigProto(gpu_options=tf.GPUOptions())

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # 加载weights
        print(cfg.WEIGHTS_FILE)
        if self.weights_file is not None:
            print('Restoring weights from: ', self.weights_file)
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter+1):
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images, self.net.labels: labels}
            # step 1-9: 训练；10：训练+tensorboard可视化； 100的整数：训练+输出变量+tensorboard可视化
            if step%self.summary_iter==0:   # 10
                if step%(self.summary_iter*10)==0:   # 100
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(fetches=[self.summary_op, self.net.total_loss, self.train_op],feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = '''{} Epoch: {}, step: {}, learning rate: {}, loss:{:5.3f}, speed:{:.3f}s/iter,
                                 load:{:.3f}s/iter, remain:{}'''.format(
                        datetime.datetime.now().strftime('%Y%m-%d %H:%M:%S'),self.data.epoch, int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss, train_timer.average_time, load_timer.average_time, train_timer.remain(step, self.max_iter)
                    )
                    print(log_str)
                # step是10的整数倍
                # tensorboard横坐标为step，step的值为10的整数倍
                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run([self.summary_op, self.train_op], feed_dict=feed_dict)
                    train_timer.toc()
                # 每迭代10次进行一次tensorboa可视化
                self.writer.add_summary(summary_str, step)
                # step是1-9 的整数倍
            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

                # 保存模型weights
            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)

    # 将config.py的参数存入config.txt
    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'congig.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')
    cfg.WEIGHT_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main(args):
    if args.gpu is not None:
        cfg.GPU = args.gpu
    if args.data_dir!=cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    yolo = YOLONet()
    pascal = pascal_voc('train')

    solver = Solver(yolo, pascal)

    print('Starting training ...')
    solver.train()
    print('Done training.')

def parse_argument():
    # tf.app.flags.DEFINE_string()   # 用tf自带的命令定义参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='YOLO_small.ckpt', type=str)   # data文件夹以及ckpt文件都是提前准备好的
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    return parser.parse_args()


if __name__=='__main__':
    # python train.py --weights YOLO_small.ckpt --gpu 0
    main(parse_argument())










