import numpy as np

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('stsb-roberta-large')

def fuzzify(s, u):
    """
    Sentence fuzzifier.
    Computes membership vector for the sentence S with respect to the
    universe U
    :param s: list of word embeddings for the sentence
    :param u: the universe matrix U with shape (K, d)
    :return: membership vectors for the sentence
    """
    f_s = np.dot(s, u.T)
    m_s = np.max(f_s, axis=0)
    m_s = np.maximum(m_s, 0, m_s)
    return m_s


def dynamax_jaccard(x, y):
    """
    DynaMax-Jaccard similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = fuzzify(x, u)
    m_y = fuzzify(y, u)

    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))
    return m_inter / m_union

# Two lists of sentences
sentences1 = ["AI-powered Learning Platform. 1 . Consolidate learning content and activity into a single platform 2 . Create , organize , track and distribute learning content 3 . Suggest learn paths/content accord to business objective or after study learner habit 4 . Integrate with Massive Open Online Courses ( MOOCs ) , e.g . edX , coursera and other external source 5 . Automatically mine relevant learn content from internal and external source and keep it up to date 6 . Analyse figure and generate report 7 . Promote a collaborative learning environment where learner discuss and resolve issue arise from the learning content. AI-powered learning platform for all staff"]
sentences2 = ["AI and IT enhanced learning platform. Immersive Education Academy be an education technology company base in Hong Kong . Over the past 8 year we have develop and streamline a custom build AI and IT enhance learn management system that be capable of automatically create personalize content and learning plan for teacher and student . This platform be be use mainly for English learning but it can be use for other subject application as well . There be three main objective that we focus on : - Increased engagement for student - Personalized learning plan for student - Reduced workload for teacher and student Our company 's technology achieve this by collect large quantity of learn behavior data from student and then identify key data correlation which be then use to create personalized learn model for student . The recommended plan be provide to the teacher . The expected result be an increase in student learning efficiency and performance outcome . This project be in support the government ‚Äô s Smart City initiative . Specifically , the main policy list below : ( a ) make use of innovation and technology ( I & T ) to address urban challenge , enhance the effectiveness of city management and improve people ‚Äô s quality of living as well as Hong Kong ‚Äô s sustainability , efficiency and safety ; ( b ) enhance Hong Kong ‚Äô s attractiveness to global business and talent ; and ( c ) inspire continuous city innovation and sustainable economic development . Our project have synergy with the above policy direction because our project incorporate the development of AI and IT enhance technology that be mean for use in the education sector . The project also aim to use the data that be collect to improve the overall efficiency of education by build individualized learning model which can be use to support teacher and student at different education level . Moreover , while current learn subject use this model be mainly relate to English education , the learning platform and model can be expand adapt to other learning subject . Within the Smart City initiative , the policy relate to Smart People be most relevant as this project can be use to cultivate a learning environment for student that incorporate AI and IT learning tool , and can further support the development of local talent by mean of English language learn - far enhance the city international reputation and workforce readiness with the multinational company base in Hong Kong .. As mention in the Solution Description section , there be inherent synergy with Hong Kong 's Smart City initiative . The objective of our technology solution be both practical , scalable and multifunctional . - Increased engagement for student - Personalized learning plan for student - Reduced workload for teacher and student These be commonly accepted goal that both the public and the government desire . Hong Kong be focus on drive innovation in order to achieve good and more efficient learn outcome . Our solution do not replace the traditional method of learn but improve it by create a non intrusive blend model of learn . Therefore the risk be low ."]

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

#Compute dynamax similarity
dynamax_scores = dynamax_jaccard(embeddings1, embeddings2)

#Output the pairs with cos score
print(cosine_scores)

#Output the pairs with dynamax score
print(dynamax_scores)