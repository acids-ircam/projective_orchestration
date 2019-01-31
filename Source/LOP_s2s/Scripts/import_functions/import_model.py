def import_model(name):
	if name=="mlp_K":
		from LOP.Models.Real_time.Baseline.mlp_K import MLP_K as Model
	elif name=="LSTM_plugged_base":
		from LOP.Models.Real_time.Additive.LSTM_plugged_base import LSTM_plugged_base as Model
	elif name=="linear_regression":
		from LOP.Models.Real_time.Baseline.linear_regression import Linear_regression as Model

	# NADEs models
	elif name=="Odnade_mlp":
		from LOP.Models.NADE.odnade_mlp import Odnade_mlp as Model
	elif name=="Odnade_rnn":
		from LOP.Models.NADE.odnade_rnn import Odnade_rnn as Model

	# Correction model
	elif name=='correction_0':
		from LOP.Models.Correction.correction_0 import correction_0 as Model		

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

	else:
		raise Exception("Not a model name")
	return Model
