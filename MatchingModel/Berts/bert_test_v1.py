from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

#Sentences are encoded by calling model.encode()
bn1 = "Traditional Closed Circuit Television (CCTV) cameras are installed at strategic locations in Hong Kong to allow the Transport Department Emergency Transport Coordination Centre (ETCC) to monitor traffic condition and take remedial actions in case of emergency. There are around 1,000 fixed or pan-tilt-zoom cameras with different years of installation. At present, monitoring of camera images are performed manually by ETCC operators.  Proof of concept exercise would like to be conducted to study how Artificial Intelligence technologies could automatically monitor CCTV video and alert ETCC operators on any abnormal traffic condition."
it1 = "This technology can be used to replace traditional data statistics methods, and suitable for environments that are prolonged, dense, and heavily overlapping. Users can accurately grasp real-time detection/recognition results through Data Visualization reports. Which helps with security monitoring, multiple object/object tracking, market statistics, smart city planning, etc."
it2= "Blockchain technology is not only a platform on which the mass of new data derived from smart cities can be safely stored and accessed by those who should have access to it. The chain also may serve as the interoperable platform that gives residents of smart cities greater say in the decisions affecting their hyper-local communities, from budgeting to elections, etc. It may also serve as a reputation management tool, as these cities tend to be chock-full of citizens who demand a certain standard from individuals and businesses when it comes to communal and environmental care."


embn1 = model.encode(bn1)
# emb2 = model.encode(it2)

print("embn1:", len(embn1), embn1)

cos_sim = util.pytorch_cos_sim(emb1, emb2)
print("Cosine-Similarity:", cos_sim)