from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import TimeDistributed, Activation, Concatenate
from tensorflow.keras.models import Model

from ...utils.model_utils.metrics_losses import smape_k
from ...utils.model_utils.abstract_models import _AbstractHyperEstimator


class TimeDistributedMLP(_AbstractHyperEstimator):
    """
    """
    def __init__(self, n_features, max_activities, prob=False, model_tag=None):
        """
        """
        self.n_features = n_features
        self.max_activities = max_activities
        if model_tag is None:
            self.model_tag = 'MLP'
        else:
            self.model_tag = model_tag
        self.prob = prob

    def build(self, hp):
        """
        """
        chosen_optimizer = hp.Choice(
            name='{}_optimizer'.format(self.model_tag),
            values=['rmsprop', 'adam']
        )
        self.dropout_spatial = hp.Boolean(
            name='{}_dropout_spatial'.format(self.model_tag)
        )
        self.dropout_rate = hp.Float(
            min_value=0.0,
            max_value=0.4,
            step=0.1,
            name='{}_dropout_rate'.format(self.model_tag)
        )

        # I LEVEL INPUTS
        feat_input = Input(
            shape=(None, self.n_features),
            name='features_input'
        )
        cont_input = Input(
            shape=(None, ),
            name='context_input'
        )
        area_input = Input(
            shape=(None,),
            name='area_input'
        )
        hour_input = Input(
            shape=(None, ),
            name='hours_input'
        )
        month_input = Input(
            shape=(None, ),
            name='month'
        )
        day_week_input = Input(
            shape=(None, ),
            name='days_week_input'
        )
        day_month_input = Input(
            shape=(None, ),
            name='days_month_input'
        )
        events_input = Input(
            shape=(None, self.max_activities),
            name='events_input'
        )
        freq_input = Input(
            shape=(None, self.max_activities),
            name='freq_input'
        )

        #######################################################################

        cont_embedding = self._generate_embedding_block(
            hp=hp,
            input_tensor=cont_input,
            input_dim=10,
            tag='context'
        )
        area_embedding = self._generate_embedding_block(
            hp=hp,
            input_tensor=area_input,
            input_dim=250,
            tag='area'
        )
        hour_embedding = self._generate_embedding_block(
            hp=hp,
            input_tensor=hour_input,
            input_dim=25,
            tag='hours'
        )
        month_embedding = self._generate_embedding_block(
            hp=hp,
            input_tensor=month_input,
            input_dim=13,
            tag='months'
        )
        day_week_embedding = self._generate_embedding_block(
            hp=hp,
            input_tensor=day_week_input,
            input_dim=8,
            tag='days_week'
        )
        day_month_embedding = self._generate_embedding_block(
            hp=hp,
            input_tensor=day_month_input,
            input_dim=32,
            tag='days_month'
        )
        events_embedding = self._events_embedding_block(
            hp=hp,
            input_events=events_input,
            input_freq=freq_input,
            input_dim=40,
            tag='events_embed',
            max_activities=self.max_activities
        )

        #######################################################################

        features_env = Concatenate(
            name='features_env_concatenation'
        )(
            [
             area_embedding,
             cont_embedding,
             hour_embedding,
             month_embedding,
             day_week_embedding,
             day_month_embedding,
            ]
        )
        features_env = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=features_env,
            tag='features_env',
            prob=self.prob,
            max_layers=10
        )

        #######################################################################

        features_event = Concatenate(
            name='features_event_concatenation'
        )(
            [
             cont_embedding,
             events_embedding
            ]
        )
        features_event = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=features_event,
            tag='features_event',
            prob=self.prob,
            max_layers=10
        )

        #######################################################################

        features_beha = Concatenate(
            name='features_beha_concatenation'
        )(
            [
             cont_embedding,
             feat_input
            ]
        )
        features_beha = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=features_beha,
            tag='features_beha',
            prob=self.prob,
            max_layers=10
        )

        #######################################################################

        features = Concatenate(
            name='features_concatenation'
        )(
            [
             features_env,
             features_event,
             features_beha
            ]
        )

        # DENSE BLOCK
        dense = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=features,
            tag='global_features',
            prob=self.prob,
            max_layers=20
        )

        # III LEVEL ESTIMATORS
        # ABSENCE ESTIMATION
        absence = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='absence',
            prob=self.prob,
            max_layers=5
        )
        absence = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_absence_td'
        )(absence)
        absence = Activation(
            'relu',
            name='output_absence_act'
        )(absence)

        # ACTIVE TIME ESTIMATION
        active = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='active',
            prob=self.prob,
            max_layers=10
        )
        active = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_active_td'
        )(active)
        active = Activation(
            'relu',
            name='output_active_act'
        )(active)

        # SESSION TIME ESTIMATION
        sess_time = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='sess_time',
            prob=self.prob,
            max_layers=10
        )
        sess_time = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_sess_time_td'
        )(sess_time)
        sess_time = Activation(
            'relu',
            name='output_sess_time_act'
        )(sess_time)

        # ACTIVITY ESTIMATION
        activity = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='activity',
            prob=self.prob,
            max_layers=10
        )
        activity = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_activity_td'
        )(activity)
        activity = Activation(
            'relu',
            name='output_activity_act'
        )(activity)

        # SESSIONS ESTIMATION
        sess = self._generate_fully_connected_block(
            hp=hp,
            input_tensor=dense,
            tag='sess',
            prob=self.prob,
            max_layers=10
        )
        sess = TimeDistributed(
            Dense(
                units=1
            ),
            name='output_sess_td'
        )(sess)
        sess = Activation(
            'relu',
            name='output_sess_act'
        )(sess)

        model = Model(
            inputs=[
                feat_input,
                cont_input,
                area_input,
                hour_input,
                month_input,
                day_week_input,
                day_month_input,
                events_input,
                freq_input
            ],
            outputs=[
                absence,
                active,
                sess_time,
                activity,
                sess
            ]
        )
        model.compile(
            optimizer=chosen_optimizer,
            loss={
                'output_absence_act': smape_k,
                'output_active_act': smape_k,
                'output_sess_time_act': smape_k,
                'output_activity_act': smape_k,
                'output_sess_act': smape_k
            },
            metrics={
                'output_absence_act': smape_k,
                'output_active_act': smape_k,
                'output_sess_time_act': smape_k,
                'output_activity_act': smape_k,
                'output_sess_act': smape_k
            }
        )
        return model
