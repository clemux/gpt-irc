import json
import numpy as np
import os
import random
import tensorflow as tf

import sopel.module, sopel.tools

import encoder, model, sample

tf.logging.set_verbosity('ERROR')

transformer = None

class Transformer:
    def __init__(self, model_name='117M', seed=None, nsamples=1,
                 batch_size=1, length=10, temperature=0.7, top_k=0,
                 nb_lines=1):
        self.model_name = model_name
        self.seed = seed
        self.nsamples = nsamples
        self.batch_size = 1
        self.length = 100
        self.temperature = temperature
        self.top_k = top_k
        self.nb_lines = nb_lines

        self.session = None

    def setup(self):
        if self.batch_size is None:
            self.batch_size = 1
        assert self.nsamples % self.batch_size == 0

        self.enc = encoder.get_encoder(self.model_name)
        hparams = model.default_hparams()
        with open(os.path.join('models', self.model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = hparams.n_ctx // 2
        elif self.length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        graph=tf.Graph()
        saver = None
        with graph.as_default():
            assert graph is tf.get_default_graph()
            self.context = tf.placeholder(tf.int32, [self.batch_size, None])
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)
            self.output = sample.sample_sequence(
                hparams=hparams, length=self.length,
                context=self.context,
                batch_size=self.batch_size,
                temperature=self.temperature,
                top_k=self.top_k
            )
            saver = tf.train.Saver()

        self.session = tf.Session(graph=graph)
        with self.session.as_default():
            ckpt = tf.train.latest_checkpoint(os.path.join('models', self.model_name))
            saver.restore(self.session, ckpt)


    def _interact(self, msg):
        context_tokens = self.enc.encode(msg)
        generated = 0
        with self.session.as_default():
            for _ in range(self.nsamples // self.batch_size):
                out = self.session.run(self.output, feed_dict={
                    self.context: [context_tokens for _ in range(self.batch_size)]
                })[:, len(context_tokens):]
                for i in range(self.batch_size):
                    generated += 1
                    text = self.enc.decode(out[i])
                    # FIXME: ugly hack to get only one line
                    # (model finetuned on IRC logs)
                    return '\n'.join(text.split('\n')[:self.nb_lines])

        return None


    def predict(self, msg):
        return self._interact(msg)

def setup(bot):
    global transformer
    # FIXME: put model_name, length, temperature in sopel configuration file
    # model_name: name of the folder containing the model, in 
    transformer = Transformer(
        model_name='tdl-gpu-3',
        length=20,
        nb_lines=1,
        temperature=0.8)
    transformer.setup()

    bot.memory['jobs'] = sopel.tools.SopelMemory()
    bot.memory['jobs']['count'] = 0

def shutdown(bot):
    transformer.session.close()

@sopel.module.commands('complete')
def complete(bot, trigger):
    bot.say(transformer.predict(trigger.group(2)))

@sopel.module.commands('comp5')
def comp5(bot, trigger):
    n = 5
    jid = bot.memory['jobs']['count'] + 1
    bot.memory['jobs']['count'] = jid
    bot.say('[{}] completing "{}", {} times, for {}'.format(
        jid, trigger.group(2), n, trigger.nick)
    )
    for i in range(0, n):
        bot.say("[{}.{}] {}".format(jid, i, transformer.predict(trigger.group(2))))

@sopel.module.commands('context')
def context(bot, trigger):
    context = trigger.group(2).split('./#@')
    context = '\n'.join(context)
    bot.say(transformer.predict(context))

# FIXME: DRY
@sopel.module.commands('context5')
def context5(bot, trigger):
    n = 5
    jid = bot.memory['jobs']['count'] + 1
    bot.memory['jobs']['count'] = jid

    lines = trigger.group(2).split('./#@')

    bot.say('[{}] completing {} lines, {} times, for {}'.format(
        jid,
        len(lines),
        n,
        trigger.nick,
    ))

    context = '\n'.join(lines)
    for i in range(0, n):
        bot.say("[{}.{}] {}".format(jid, i, transformer.predict(context)))
