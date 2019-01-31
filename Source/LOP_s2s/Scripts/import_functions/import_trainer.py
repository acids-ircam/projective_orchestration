def import_trainer(name, model_parameters, parameters):
    # Trainer
    if name == 'standard_trainer':
        from LOP.Scripts.standard_learning.standard_trainer import Standard_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], 'debug': parameters["debug"]}
    elif name == 'NADE_trainer':
        from LOP.Scripts.NADE_learning.NADE_trainer import NADE_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], 'num_ordering': model_parameters["num_ordering"], 'debug': parameters["debug"]}
    elif name == 'NADE_Gibbs_trainer':
        from LOP.Scripts.NADE_learning.NADE_Gibbs_trainer import NADE_Gibbs_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], 'num_ordering': model_parameters["num_ordering"], 'debug': parameters["debug"]}
    elif name == 'correct_trainer':
        from LOP.Scripts.Correct_learning.correct_trainer import correct_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], "mean_iteration_per_note": model_parameters["mean_iteration_per_note"], 'debug': parameters["debug"]}
    elif name == 'orchSeq_trainer':
        from LOP.Scripts.standard_learning.orchSeq_trainer import orchSeq_trainer as Trainer
        kwargs_trainer = {"max_length_sequence": parameters["max_length_sequence"], "n_instru": parameters["n_instru"]}
    else:
        raise Exception("Undefined trainer")
    trainer = Trainer(**kwargs_trainer)
    return trainer