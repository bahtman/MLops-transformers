from azureml.core import Workspace, Environment, Experiment, ScriptRunConfig, experiment

ws = Workspace.from_config("src/AML/config.json")
experiment = Experiment(ws,"InitialExperiment")
myenv = Environment.from_pip_requirements(name= "myenv", file_path="requirements.txt")
cluster = ws.compute_targets["StudieCompute"]
config = ScriptRunConfig(source_directory='.',
                            script='src/models/train_model.py',
                            compute_target=cluster,
                            environment=myenv)
run = experiment.submit(config)
run.wait_for_completion()
run.register_model( model_name="classification_model",
                    model_path="src/models/model.pth",
                    description="The classification model",
                    )