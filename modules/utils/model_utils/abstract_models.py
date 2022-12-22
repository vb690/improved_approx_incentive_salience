import time

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Activation, LeakyReLU, ReLU, ELU
from tensorflow.keras.layers import Dropout, SpatialDropout1D, Concatenate

from tensorflow.keras.utils import plot_model

from kerastuner import HyperModel

from tensorflow.keras.models import Model


class _AbstractHyperEstimator(HyperModel):

    ACTIVATIONS = {"relu": ReLU, "elu": ELU, "lelu": LeakyReLU}

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_model"]
        return state

    def _generate_embedding_block(
        self, hp, input_tensor, input_dim, tag, max_dim=256
    ):
        """Short summary.

        Args:
            hp (type): Description of parameter `hp`.
            input_tensor (type): Description of parameter `input_tensor`.
            input_dim (type): Description of parameter `input_dim`.
            tag (type): Description of parameter `tag`.
            max_dim (type): Description of parameter `max_dim`.

        Returns:
            type: Description of returned object.

        """
        embedding = Embedding(
            input_dim=input_dim,
            output_dim=hp.Int(
                min_value=8,
                max_value=max_dim,
                step=8,
                name=f"embedding_{tag}_output",
            ),
            input_length=None,
            name=f"embedding_layer_{tag}",
        )(input_tensor)

        embedding = SpatialDropout1D(
            self.dropout_rate, name=f"sp_dropout_{tag}"
        )(embedding)

        return embedding

    def _events_embedding_block(
        self,
        hp,
        input_events,
        input_freq,
        input_dim,
        tag,
        max_activities,
        max_dim=256,
    ):
        """Short summary.

        Args:
            hp (type): Description of parameter `hp`.
            input_events (type): Description of parameter `input_events`.
            input_freq (type): Description of parameter `input_freq`.
            input_dim (type): Description of parameter `input_dim`.
            tag (type): Description of parameter `tag`.
            max_activities (type): Description of parameter `max_activities`.
            max_dim (type): Description of parameter `max_dim`.

        Returns:
            type: Description of returned object.

        """
        event_embed = Embedding(
            input_dim=input_dim,
            output_dim=hp.Int(
                min_value=8,
                max_value=max_dim,
                step=8,
                name="embedding_{}_output".format(tag),
            ),
            input_length=None,
            name="embedding_layer_{}".format(tag),
        )(input_events)

        event_reshape = tf.reshape(
            event_embed,
            shape=[-1, max_activities, event_embed.get_shape().as_list()[-1]],
        )
        fre_reshape = tf.reshape(input_freq, shape=[-1, max_activities, 1])

        event_freq = Concatenate()([event_reshape, fre_reshape])

        event_freq = Conv1D(
            hp.Int(
                min_value=8,
                max_value=max_dim,
                step=8,
                name="conv_1D_{}_output".format(tag),
            ),
            kernel_size=2,
            activation="relu",
            name="conv_1D_{}_output".format(tag),
        )(event_freq)
        event_freq = GlobalAveragePooling1D()(event_freq)
        event_freq = tf.reshape(
            event_freq,
            shape=[
                tf.shape(input_freq)[0],
                tf.shape(input_freq)[1],
                event_freq.get_shape().as_list()[-1],
            ],
        )

        return event_freq

    def _generate_fully_connected_block(
        self, hp, input_tensor, tag, prob, max_layers=5, max_dim=256
    ):
        """Short summary.

        Args:
            hp (type): Description of parameter `hp`.
            input_tensor (type): Description of parameter `input_tensor`.
            tag (type): Description of parameter `tag`.
            prob (type): Description of parameter `prob`.
            max_layers (type): Description of parameter `max_layers`.
            max_dim (type): Description of parameter `max_dim`.

        Returns:
            type: Description of returned object.

        """
        layers = hp.Int(
            min_value=1,
            max_value=max_layers,
            name="dense_layers_{}".format(tag),
        )
        for layer in range(layers):

            if layer == 0:
                fully_connected = Dense(
                    units=hp.Int(
                        min_value=8,
                        max_value=max_dim,
                        step=8,
                        name="dense_units_{}_{}".format(layer, tag),
                    ),
                    name="{}_dense_layer_{}".format(layer, tag),
                )(input_tensor)
            else:
                fully_connected = Dense(
                    units=hp.Int(
                        min_value=8,
                        max_value=max_dim,
                        step=8,
                        name="dense_units_{}_{}".format(layer, tag),
                    ),
                    name="{}_dense_layer_{}".format(layer, tag),
                )(fully_connected)

            chos_act = hp.Choice(
                values=["elu", "relu", "lelu"],
                name="dense_activation_{}_{}".format(layer, tag),
            )
            fully_connected = Activation(
                self.ACTIVATIONS[chos_act](),
                name="{}_{}_activation_dense_layer_{}".format(
                    layer, chos_act, tag
                ),
            )(fully_connected)

            if self.dropout_spatial:
                fully_connected = SpatialDropout1D(
                    self.dropout_rate,
                    name="sp_dropout_{}_{}".format(layer, tag),
                )(fully_connected)
            else:
                fully_connected = Dropout(
                    self.dropout_rate, name="dropout_{}_{}".format(layer, tag)
                )(fully_connected)

        return fully_connected

    def _generate_recurrent_block(
        self, hp, input_tensor, tag, max_layers=2, max_dim=256
    ):
        """Short summary.

        Args:
            hp (type): Description of parameter `hp`.
            input_tensor (type): Description of parameter `input_tensor`.
            tag (type): Description of parameter `tag`.
            max_layers (type): Description of parameter `max_layers`.
            max_dim (type): Description of parameter `max_dim`.

        Returns:
            type: Description of returned object.

        """
        layers = hp.Int(
            min_value=1, max_value=max_layers, name="lstm_layers_{}".format(tag)
        )
        for layer in range(layers):

            if layer == 0:
                recurrent = LSTM(
                    units=hp.Int(
                        min_value=8,
                        max_value=max_dim,
                        step=8,
                        name="lstm_units_{}_{}".format(layer, tag),
                    ),
                    return_sequences=True,
                    name="{}_lstm_layer_{}".format(layer, tag),
                )(input_tensor)
            else:
                recurrent = LSTM(
                    units=hp.Int(
                        min_value=8,
                        max_value=max_dim,
                        step=8,
                        name="lstm_units_{}_{}".format(layer, tag),
                    ),
                    return_sequences=True,
                    name="{}_lstm_layer_{}".format(layer, tag),
                )(recurrent)

            recurrent = SpatialDropout1D(
                self.dropout_rate, name="{}_sp_dropout_{}".format(layer, tag)
            )(recurrent)

        return recurrent

    def get_para_count(self):
        """Short summary.

        Returns:
            type: Description of returned object.

        """
        num_parameters = self.n_parameters
        return num_parameters

    def get_model(self):
        """Short summary.

        Returns:
            type: Description of returned object.

        """
        model = self._model
        return model

    def get_model_tag(self):
        """Short summary.

        Returns:
            type: Description of returned object.

        """
        model_tag = self.model_tag
        return model_tag

    def get_fitting_time(self):
        """Short summary.

        Returns:
            type: Description of returned object.

        """
        fitting_time = self.fitting_time
        return fitting_time

    def get_n_epochs(self):
        """Short summary.

        Returns:
            type: Description of returned object.

        """
        n_epochs = self.n_epochs
        return n_epochs

    def get_encoder(self, out_layer, inp_layers=None):
        """Short summary.

        Args:
            out_layer (type): Description of parameter `out_layer`.
            inp_layers (type): Description of parameter `inp_layers`.

        Returns:
            type: Description of returned object.

        """
        if inp_layers is not None:
            inputs = []
            for inp in self._model.inputs:

                if inp.name in inp_layers:
                    inputs.append(inp)
        else:
            inputs = self._model.inputs

        out = self._model.get_layer(out_layer)
        encoder = Model(inputs, out.output)
        return encoder

    def set_model(self, model):
        """Short summary.

        Args:
            model (type): Description of parameter `model`.

        Returns:
            type: Description of returned object.

        """
        setattr(self, "_model", model)
        setattr(self, "n_parameters", model.count_params())

    def tune(
        self,
        tuner,
        generator,
        validation_data,
        epochs=30,
        callbacks=None,
        verbose=2,
        **kwargs
    ):
        """Short summary.

        Args:
            tuner (type): Description of parameter `tuner`.
            generator (type): Description of parameter `generator`.
            validation_data (type): Description of parameter `validation_data`.
            epochs (type): Description of parameter `epochs`.
            callbacks (type): Description of parameter `callbacks`.
            verbose (type): Description of parameter `verbose`.
            **kwargs (type): Description of parameter `**kwargs`.

        Returns:
            type: Description of returned object.

        """
        tuner_obj = tuner(hypermodel=self.build, **kwargs)

        tuner_obj.search(
            generator,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
        )
        tuner_obj.results_summary()
        model = tuner_obj.get_best_models()[0]

        plot_path = "results\\figures\\architectures"
        plot_model(
            model, to_file="{}\\{}.png".format(plot_path, self.model_tag)
        )
        self.set_model(model)

    def fit(self, **kwargs):
        """Short summary.

        Args:
            **kwargs (type): Description of parameter `**kwargs`.

        Returns:
            type: Description of returned object.

        """
        start = time.time()
        history = self._model.fit(**kwargs)
        end = time.time()
        setattr(self, "n_epochs", len(history.history["loss"]))
        setattr(self, "fitting_time", end - start)
        return history

    def predict(self, **kwargs):
        """Short summary.

        Args:
            **kwargs (type): Description of parameter `**kwargs`.

        Returns:
            type: Description of returned object.

        """
        prediction = self._model.predict(**kwargs)
        return prediction

    def predict_with_uncertainty(self, X_test, y_test, n_iter, **kwargs):
        """Short summary.

        Args:
            X_test (type): Description of parameter `X_test`.
            y_test (type): Description of parameter `y_test`.
            n_iter (type): Description of parameter `n_iter`.
            **kwargs (type): Description of parameter `**kwargs`.

        Returns:
            type: Description of returned object.

        """
        if self.prob is not True or self.hp_schema["dropout"] == 0:
            raise ValueError("Non-probabilistic is used")
        predictions = np.empty(shape=(y_test.shape[0], n_iter, y_test.shape[1]))
        for iter in range(n_iter):

            prediction = self._model.predict(X_test, **kwargs)
            predictions[:, iter, :] = prediction

        return predictions
