from azureml.core.webservice import LocalWebservice
from azureml.core import Environment, Workspace
from azureml.core.model import InferenceConfig, Model
import requests
import json
ws = Workspace.from_config("src/AML/config.json")
myenv = Environment.from_pip_requirements(name= "myenv", file_path="requirements.txt")
dummy_inference_config = InferenceConfig(
    environment=myenv,
    source_directory="./src",
    entry_script="./AML/score.py",
)
deployment_config = LocalWebservice.deploy_configuration(port=1234)
model = Model(ws,"Classification_model")
service = Model.deploy(
    ws,
    "myservice",
    [model],
    dummy_inference_config,
    deployment_config,
    overwrite=True,
)
service.wait_for_deployment(show_output=True)

uri = service.scoring_uri
headers = {"Content-Type": "application/json"}
while True:
    sms = input("Type a sms: ")
    data = {
        "sms": sms
        #"sms": "We are using the ”Bert” framework to convert the text messages into numbers.We will then use the numbers as inputs to a simple feed forward network, whichwe will train in order to classify the messaseges as ”SPAM ” or ”HAM”.",
        #"sms": "Deaar Anton see your prizes here: http://ihobem.com/f2GojS5"
    }
    data = json.dumps(data)
    response = requests.post(uri, data=data, headers=headers)
    print(response.json()["output"])
