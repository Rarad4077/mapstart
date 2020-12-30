#HuggingFace Summerizer

from transformers import pipeline
summarizer = pipeline("summarization")
ARTICLE = """- Free from VR headsets, shared immersive experience for multi-user communication & collaboration
- First-person-view navigation in reallife 1-to-1 scale
- 3D preview future construction design: Simulate initial design concepts to identify and fix potential problems at early stage
"""

print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

#method 2
from transformers import AutoModelWithLMHead, AutoTokenizer
model = AutoModelWithLMHead.from_pretrained("t5-base", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("t5-base")
# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="pt", max_length=512)
outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)