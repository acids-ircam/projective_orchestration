def import_trainer(name, model_parameters, parameters):
    # Trainer
    if name == 'standard_trainer':
        from LOP.Scripts.trainers.standard_trainer import Standard_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], 'debug': parameters["debug"]}
    
    elif name == 'NADE_trainer':
        from LOP.Scripts.trainers.NADE_learning.NADE_trainer import NADE_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], 'num_ordering': model_parameters["num_ordering"], 'debug': parameters["debug"]}
    elif name == 'NADE_Gibbs_trainer':
        from LOP.Scripts.trainers.NADE_learning.NADE_Gibbs_trainer import NADE_Gibbs_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], 'num_ordering': model_parameters["num_ordering"], 'debug': parameters["debug"]}
    elif name == 'NADE_informed_trainer':
        from LOP.Scripts.trainers.NADE_learning.NADE_informed_trainer import NADE_informed_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], 'num_ordering': model_parameters["num_ordering"], 'debug': parameters["debug"],
            'orch_to_orch': parameters['mask_inter_orch_NADE'],
            'piano_to_orch': parameters['mask_piano_orch_NADE']
            }
        
    elif name == 'correct_trainer':
        from LOP.Scripts.trainers.correct_learning.correct_trainer import correct_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], "mean_iteration_per_note": model_parameters["mean_iteration_per_note"], 'debug': parameters["debug"]}
    elif name == 'correct_trainer_balanced':
        from LOP.Scripts.trainers.correct_learning.correct_trainer_balanced import correct_trainer_balanced as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], "mean_iteration_per_note": model_parameters["mean_iteration_per_note"], 'debug': parameters["debug"]}
    elif name == 'correct_trainer_balanced_NADE_style':
        from LOP.Scripts.trainers.correct_learning.correct_trainer_balanced_NADE_style import correct_trainer_balanced_NADE_style as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], "mean_iteration_per_note": model_parameters["mean_iteration_per_note"], 'debug': parameters["debug"]}
    
    elif name == 'orchSeq_trainer':
        from LOP.Scripts.trainers.orchSeq_trainer import orchSeq_trainer as Trainer
        kwargs_trainer = {"max_length_sequence": parameters["max_length_sequence"], "n_instru": parameters["n_instru"]}

    elif name == 'energy_based':
        from LOP.Scripts.trainers.energy_trainer import Energy_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], 'debug': parameters["debug"]}

    elif name == 'sequence_trainer':
        from LOP.Scripts.trainers.sequence_trainer import Sequence_trainer as Trainer
        kwargs_trainer = {'temporal_order': model_parameters["temporal_order"], 'debug': parameters["debug"]}

    else:
        raise Exception("Undefined trainer")
    trainer = Trainer(**kwargs_trainer)
    return trainer