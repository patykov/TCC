import cntk as C
import os


def define_learner(model, lr_per_mb, mb_size, momentum=0.9, l2_reg_weight=0.0001, distributed=True):
	lr_schedule = C.learners.learning_parameter_schedule(lr_per_mb, mb_size, mb_size)
	mm_schedule = C.learners.momentum_schedule(momentum)
	
	learner = C.learners.momentum_sgd(model.parameters, lr_schedule, mm_schedule, 
								unit_gain=False, l2_regularization_weight=l2_reg_weight)
	if distributed:
		learner = C.data_parallel_distributed_learner(learner, 
													num_quantization_bits=32, 
													distributed_after=0)
	return learner


def train(model_name, network, train_reader, input_map, learner, mb_size, epoch_size, 
		   max_epochs, output_dir, log_file, profiling=False, distributed=True):

	# Printer
	progress_printer = C.logging.ProgressPrinter(tag='Training', log_to_file=log_file, 
										num_epochs=max_epochs, gen_heartbeat=True, 
										rank=C.Communicator.rank() if distributed else None)
	
	# Trainer object
	trainer = C.Trainer(network['model'], (network['ce'], network['pe']), learner, progress_printer)
	
	if profiling:
		C.debbuging.start_profiler(dir=output_dir, sync_gpu=True)

	C.train.training_session(
		trainer=trainer, mb_source=train_reader,
		model_inputs_to_streams=input_map,
		mb_size=mb_size,
		progress_frequency=epoch_size,
		checkpoint_config=C.train.CheckpointConfig(filename=os.path.join(output_dir, model_name), 
													restore=True, frequency=int(epoch_size/2))
	).train()

	if profiling:
		C.debbuging.stop_profiler()
	
	if learner.communicator().is_main():
		trainer.model.save_model(os.path.join(output_dir, "{}.model".format(model_name)))
