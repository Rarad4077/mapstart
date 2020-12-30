from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

#Sentences are encoded by calling model.encode()
bn1 = "blockchain"
it1 = "Founded in 2001, Shanghai Xiaoi Robot Technology Co. Ltd is a leading AI company that specializes in multilingual natural language processing, deep semantic interaction, speech recognition, machine learning and other cognitive intelligence technologies. Its self-patented NLP-powered Chatbot Solutions possess the following capabilities. We have ample use cases across various business domains and public sectors, with solid track record in deploying solutions and applications to banking & finance, insurance, healthcare, education, transportation, e-commerce, FMCG, utilities, infrastructure and government projects etc."
it2= "Blockchain technology is not only a platform on which the mass of new data derived from smart cities can be safely stored and accessed by those who should have access to it. The chain also may serve as the interoperable platform that gives residents of smart cities greater say in the decisions affecting their hyper-local communities, from budgeting to elections, etc. It may also serve as a reputation management tool, as these cities tend to be chock-full of citizens who demand a certain standard from individuals and businesses when it comes to communal and environmental care."


emb1 = model.encode(bn1)
emb2 = model.encode(it2)
cos_sim = util.pytorch_cos_sim(emb1, emb2)
print("emb1", len(emb2))
print("Cosine-Similarity:", cos_sim.item(), type(cos_sim.item()))