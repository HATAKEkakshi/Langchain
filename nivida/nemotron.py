from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="nvidia/nemotron-4-340b-reward",
  messages=[{"role":"user","content":"I am going to Paris, what should I see?"},{"role":"assistant","content":"Ah, Paris, the City of Light! There are so many amazing things to see and do in this beautiful city ..."}],
)
print(completion)
