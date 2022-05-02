from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Dense

from ...utils.model_utils.abstract_models import _AbstractHyperEstimator
from ...utils.model_utils.metrics_losses import smape_k


class RecurrentModel(_AbstractHyperEstimator):
    """ """

    def __init__(
        self,
        n_features,
        max_activities,
        prob=False,
        model_tag=None,
        adjust_for_env=True,
        adjust_for_events=True,
    ):
        """Short summary.

        Args:
            n_features (type): Description of parameter `n_features`.
            max_activities (type): Description of parameter `max_activities`.
            prob (type): Description of parameter `prob`.
            model_tag (type): Description of parameter `model_tag`.
            adjust_for_env (type): Description of parameter `adjust_for_env`.
            adjust_for_events (type): Description of parameter `adjust_for_events`.

        Returns:
            type: Description of returned object.

        """
        self.n_features = n_features
        self.max_activities = max_activities
        if model_tag is None:
            self.model_tag = "RNN"
        else:
            self.model_tag = model_tag
        self.prob = prob
        self.adjust_for_env = adjust_for_env
        self.adjust_for_events = adjust_for_events

    def build(self, hp):
        """Short summary.

        Args:
            hp (type): Description of parameter `hp`.

        Returns:
            type: Description of returned object.

        """
        chosen_optimizer = hp.Choice(
            name="{}_optimizer".format(self.model_tag),
            values=["rmsprop", "adam"],
        )
        self.dropout_rate = hp.Float(
            min_value=0.0,
            max_value=0.4,
            step=0.05,
            name="{}_dropout_rate".format(self.model_tag),
        )
        self.dropout_spatial = True

        # I LEVEL INPUTS
        feat_input = Input(shape=(None, self.n_features), name="features_input")
        cont_input = Input(shape=(None,), name="context_input")

        cont_embedding = self._generate_embedding_block(
            hp=hp, input_tensor=cont_input, input_dim=7, tag="context"
        )
        model_input_tensors = [feat_input, cont_input]

        # OPTIONAL ENVIRONMENT ADJUSTING
        if self.adjust_for_env:
            area_input = Input(shape=(None,), name="area_input")
            hour_input = Input(shape=(None,), name="hours_input")
            month_input = Input(shape=(None,), name="month")
            day_week_input = Input(shape=(None,), name="days_week_input")
            day_month_input = Input(shape=(None,), name="days_month_input")

            model_input_tensors.extend(
                [
                    area_input,
                    hour_input,
                    month_input,
                    day_week_input,
                    day_month_input,
                ]
            )

            cont_embedding = self._generate_embedding_block(
                hp=hp, input_tensor=cont_input, input_dim=10, tag="context"
            )
            area_embedding = self._generate_embedding_block(
                hp=hp, input_tensor=area_input, input_dim=250, tag="area"
            )
            hour_embedding = self._generate_embedding_block(
                hp=hp, input_tensor=hour_input, input_dim=25, tag="hours"
            )
            month_embedding = self._generate_embedding_block(
                hp=hp, input_tensor=month_input, input_dim=13, tag="months"
            )
            day_week_embedding = self._generate_embedding_block(
                hp=hp, input_tensor=day_week_input, input_dim=8, tag="days_week"
            )
            day_month_embedding = self._generate_embedding_block(
                hp=hp,
                input_tensor=day_month_input,
                input_dim=32,
                tag="days_month",
            )

        if self.adjust_for_events:
            events_input = Input(
                shape=(None, self.max_activities), name="events_input"
            )
            freq_input = Input(
                shape=(None, self.max_activities), name="freq_input"
            )

            model_input_tensors.extend([events_input, freq_input])
            events_embedding = self._events_embedding_block(
                hp=hp,
                input_events=events_input,
                input_freq=freq_input,
                input_dim=40,
                tag="events_embed",
                max_activities=self.max_activities,
            )

        # II LEVEL FIRST LSTM
        # explicitly model the contribution of the behavioural features
        # weighted by the context
        feat_cont_concat = Concatenate(name="concat_feat_cont")(
            [feat_input, cont_embedding]
        )

        feat_recurrent = self._generate_recurrent_block(
            hp=hp, input_tensor=feat_cont_concat, tag="features", max_layers=1
        )

        cov_tensors = [feat_recurrent]
        if self.adjust_for_env:
            env = Concatenate()(
                [
                    cont_embedding,
                    area_embedding,
                    hour_embedding,
                    month_embedding,
                    day_week_embedding,
                    day_month_embedding,
                ]
            )
            env_recurrent = self._generate_recurrent_block(
                hp=hp, input_tensor=env, tag="env", max_layers=1
            )
            cov_tensors.append(env_recurrent)
        if self.adjust_for_events:
            events = Concatenate()(
                [
                    cont_embedding,
                    events_embedding,
                ]
            )
            events_recurrent = self._generate_recurrent_block(
                hp=hp, input_tensor=events, tag="events", max_layers=1
            )
            cov_tensors.append(events_recurrent)
        if self.adjust_for_env or self.adjust_for_events:
            shared = Concatenate(name="shared")(cov_tensors)

            shared_recurrent = self._generate_recurrent_block(
                hp=hp, input_tensor=shared, tag="shared", max_layers=1
            )
        else:
            shared_recurrent = feat_recurrent

        # VI LEVEL ESTIMATORS
        # create the heds for ostimating the target metrics

        # ABSENCE ESTIMATION
        absence = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag="absence",
            prob=self.prob,
            max_layers=3,
        )
        absence = Dense(units=1, name="output_absence_time_td")(absence)
        absence = Activation("relu", name="output_absence_act")(absence)

        # ACTIVE TIME ESTIMATION
        active = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag="active",
            prob=self.prob,
            max_layers=3,
        )
        active = Dense(units=1, name="output_active_td")(active)
        active = Activation("relu", name="output_active_act")(active)

        # SESSION TIME ESTIMATION
        sess_time = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag="sess_time",
            prob=self.prob,
            max_layers=3,
        )
        sess_time = Dense(units=1, name="output_sess_time_td")(sess_time)
        sess_time = Activation("relu", name="output_sess_time_act")(sess_time)

        # ACTIVITY ESTIMATION
        activity = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag="activity",
            prob=self.prob,
            max_layers=3,
        )
        activity = Dense(units=1, name="output_activity_td")(activity)
        activity = Activation("relu", name="output_activity_act")(activity)

        # SESSIONS ESTIMATION
        sess = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=shared_recurrent,
            tag="sess",
            prob=self.prob,
            max_layers=3,
        )
        sess = Dense(units=1, name="output_sess_td")(sess)
        sess = Activation("relu", name="output_sess_act")(sess)

        model = Model(
            inputs=model_input_tensors,
            outputs=[absence, active, sess_time, activity, sess],
        )
        model.compile(
            optimizer=chosen_optimizer,
            loss={
                "output_absence_act": smape_k,
                "output_active_act": smape_k,
                "output_sess_time_act": smape_k,
                "output_activity_act": smape_k,
                "output_sess_act": smape_k,
            },
            metrics={
                "output_absence_act": smape_k,
                "output_active_act": smape_k,
                "output_sess_time_act": smape_k,
                "output_activity_act": smape_k,
                "output_sess_act": smape_k,
            },
        )
        return model
