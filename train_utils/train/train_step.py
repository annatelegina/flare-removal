import tensorflow as tf

def make_train_step_PatchGAN():
    @tf.function
    def train_step_PatchGAN(input_image,
                       target,
                       generator,
                       discriminator,
                       generator_loss,
                       discriminator_loss,
                       generator_optimizer,
                       discriminator_optimizer,
                       training=True,
                       fullres=False):
        """
        Performs one train step for PatchGAN training.
        :param input_image: tf.tensor, batch of input images
        :param target: tf.tensor, batch of target images
        :param generator: keras.model, generator net
        :param discriminator: keras.model, discriminator net
        :param generator_loss: function, takes 3 args: disc_generated_output, target, gen_output;
        returns tuple (gen_total_loss, [other losses])
        :param discriminator_loss: function, takes 2 args: disc_real_output, disc_generated_output;
        returns discriminator total loss
        :param generator_optimizer: optimizer for generator
        :param discriminator_optimizer: optimizer for discriminator
        :param training:
        :return: gen total loss value, [other gen losses], disc value
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=training)

            if fullres:
                disc_real_output = discriminator([input_image[:, :, :, :3], target], training=training)
                disc_generated_output = discriminator([input_image[:, :, :, :3], gen_output], training=training)
            else:
                disc_real_output = discriminator([input_image, target], training=training)
                disc_generated_output = discriminator([input_image, gen_output], training=training)

            gen_total_loss, gen_losses = generator_loss(disc_generated_output, target, gen_output)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

        return gen_total_loss, gen_losses, disc_loss
    return train_step_PatchGAN


def make_train_step():
    @tf.function
    def train_step(input_image,
                   target,
                   generator,
                   generator_loss,
                   generator_optimizer,
                   training=True):
        """
        Performs one train step for generator only training.
        :param input_image: tf.tensor, batch of input images
        :param target: tf.tensor, batch of target images
        :param generator: keras.model, generator net
        :param generator_loss: function, takes 2 args: target, gen_output;
        returns tuple (gen_total_loss, [other losses])
        :param generator_optimizer: optimizer for generator
        :param training:
        :return:
        """
        with tf.GradientTape() as gen_tape:
            gen_output = generator(input_image, training=training)
            gen_total_loss, gen_losses = generator_loss(gen_output, target)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))

        return gen_total_loss, gen_losses
    return train_step


def make_train_step_srs():
    @tf.function
    def train_step(input_image,
                   usc_uint16_image,
                   target,
                   generator,
                   generator_loss,
                   generator_optimizer,
                   training=True):
        """
        Performs one train step for vanilla GAN training.
        :param input_image: tf.tensor, batch of input images
        :param target: tf.tensor, batch of target images
        :param generator: keras.model, generator net
        :param generator_loss: function, takes 2 args: target, gen_output;
        returns tuple (gen_total_loss, [other losses])
        :param generator_optimizer: optimizer for generator
        :param training:
        :return:
        """
        with tf.GradientTape() as gen_tape:
            gen_output = generator(input_image, training=training)
            gen_total_loss, gen_losses = generator_loss(gen_output[0], usc_uint16_image, target)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))

        return gen_total_loss, gen_losses
    return train_step

def make_train_step_srs_multi_output():
    @tf.function
    def train_step_multi_output(input_image,
                   usc_uint16_image,
                   target,
                   generator,
                   generator_loss,
                   generator_optimizer,
                   training=True):
        """
        Performs one train step for vanilla GAN training.
        :param input_image: tf.tensor, batch of input images
        :param target: tf.tensor, batch of target images
        :param generator: keras.model, generator net
        :param generator_loss: function, takes 2 args: target, gen_output;
        returns tuple (gen_total_loss, [other losses])
        :param generator_optimizer: optimizer for generator
        :param training:
        :return:
        """
        with tf.GradientTape() as gen_tape:
            gen_output = generator(input_image, training=training)
            print(len(gen_output) )
            gen_total_loss, gen_losses = generator_loss(gen_output, usc_uint16_image, target)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))

        return gen_total_loss, gen_losses
    return train_step_multi_output

