def import_model(name):
	if name=="mlp_K":
		from LOP.Models.Real_time.Additive.Baseline.mlp_K import MLP_K as Model
	elif name=="repeat":
		from LOP.Models.Real_time.Additive.Baseline.repeat import Repeat as Model
	elif name=="zeros":
		from LOP.Models.Real_time.Additive.Baseline.zeros import Zeros as Model
	elif name=="LSTM_plugged_base":
		from LOP.Models.Real_time.Additive.LSTM_plugged_base import LSTM_plugged_base as Model
	elif name=="LSTM_conv_base":
		from LOP.Models.Real_time.Additive.LSTM_conv_base import LSTM_conv_base as Model
	elif name=="linear_regression":
		from LOP.Models.Real_time.Additive.Baseline.linear_regression import Linear_regression as Model
	elif name=="piano_solo":
		from LOP.Models.Real_time.Additive.Baseline.piano_solo import Piano_solo as Model

	# Future past models
	elif name=="Future_past_piano":
		from LOP.Models.Future_past_piano.future_past_piano import Future_past_piano as Model

	elif name=="LSTM_static_bias":
		from LOP.Models.Real_time.Additive.LSTM_static_bias import LSTM_static_bias as Model

	# NADEs models
	elif name=="Odnade_mlp":
		from LOP.Models.NADE.odnade_mlp import Odnade_mlp as Model
	elif name=="Odnade_mlp_2":
		from LOP.Models.NADE.odnade_mlp_2 import Odnade_mlp_2 as Model
	elif name=="Odnade_rnn":
		from LOP.Models.NADE.odnade_rnn import Odnade_rnn as Model

	# Correction model
	elif name=='LSTM_correction_vector_wise':
		from LOP.Models.Correction_vector_wise.LSTM_correction_vector_wise import LSTM_correction_vector_wise as Model
	elif name=='correction_0':
		from LOP.Models.Correction.correction_0 import correction_0 as Model		
	elif name=="correction_NADE_style":
		from LOP.Models.Correction.correction_NADE_style import correction_NADE_style as Model		

	# Residual
	elif name=='LSTM_plugged_residual':
		from LOP.Models.Real_time.Additive.Residual.LSTM_plugged_residual import LSTM_plugged_residual as Model		

	# FiLM
	elif name=='FiLM_residual':
		from LOP.Models.Real_time.Affine.Residual.FiLM_residual import FiLM_residual as Model		
	elif name=="LSTM_affine_OrchCond":
		from LOP.Models.Real_time.Affine.LSTM_affine_OrchCond import LSTM_affine_OrchCond as Model				

	elif name=="seq2seq_0":
		from LOP.Models.Seq2seq.seq2seq_0 import seq2seq_0 as Model

	# Energy-based models
	elif name=="RBM":
		from LOP.Models.Real_time.Energy_based.RBM import RBM as Model
	elif name=="cRBM":
		from LOP.Models.Real_time.Energy_based.cRBM import cRBM as Model
	elif name=="FGcRBM":
		from LOP.Models.Real_time.Energy_based.FGcRBM import FGcRBM as Model

	# Backward models
	elif name=="LSTM_plugged_base_BW":
		from LOP.Models.Backward_real_time.LSTM_plugged_base_BW import LSTM_plugged_base_BW as Model

	elif name=="sequence":
		from LOP.Models.Sequence.sequence import Sequence as Model
	elif name=="sequence_tm1":
		from LOP.Models.Sequence.sequence_tm1 import Sequence_tm1 as Model

	else:
		raise Exception("Not a model name")

	return Model
