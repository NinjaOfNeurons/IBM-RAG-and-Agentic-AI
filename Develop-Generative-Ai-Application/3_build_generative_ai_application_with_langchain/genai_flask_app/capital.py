from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   # api_key = "<YOUR_API_KEY>" # Normally you'd put an API key here, but we've got you covered here
                  )

params = {GenTextParamsMetaNames.DECODING_METHOD: "greedy", #greedy decoding, where the model always picks the most probable next token
          GenTextParamsMetaNames.MAX_NEW_TOKENS: 100, #maximum number of tokens the LLM can generate in a response. 
          }


model = ModelInference(
    model_id='ibm/granite-3-3-8b-instruct',
    params=params,
    credentials=credentials,
    project_id="skills-network"
)

text = """
Only reply with the answer. What is the capital of Canada?
"""

print(model.generate(text)['results'][0]['generated_text'])
