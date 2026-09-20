from helper_functions import LLM_Logs
from LLM_loader import llm

@LLM_Logs
def llm_output(prompt):
    return llm.invoke(prompt)
    
prompt = """
%INSTRUCTIONS:
Use the provided pieces of context to answer the query. If you don't know the answer, just say that you don't know, don't try to make up an answer.
%CONTEXT
The Friedrich Schiller University Jena is a university rich in tradition and strong in research with a wide range of subjects. She bundles her cutting-edge research in the profile lines Light – Life – Liberty. It is closely networked with non-university research institutions, research-based companies and well-known cultural institutions. With almost 18,000 students and more than 8,600 employees, the university significantly shapes Jena's character as a cosmopolitan and future-oriented city.
%Query
How many students are there in Friedrich Schiller University Jena?
"""
llm_out = llm_output(prompt)


prompt ="""
%INSTRUCTIONS:
Use the provided pieces of context to answer the query. If you don't know the answer, just say that you don't know, don't try to make up an answer.
%CONTEXT
We surveyed a range of habitats and recorded 817 audio-files from
678 individuals of 35 bat species across Thailand between 2003 to 2009
and Xishuangbanna in China from 2017 to 2018 (Table 1). Among
them, 353 audio-files from 218 individuals of 24 species were collected
from rainforest, limestone forest, cave, and urban areas in Xishuang-
banna (China); 504 from 469 individuals of 25 species from forest and
karsts in Thailand (14 overlapped with species in China, Table 1). A
Pettersson D-240X and two M500-384 (Pettersson Elektronik, Sweden)
recording devices were used in Thailand and China as previously de-
scribed (Hughes et al., 2011). In addition 21 audio-files from 21 in-
dividuals of four species collected in Malaysia, which extracted from a
public bioacoustic database (Baker et al., 2015)
%Query
How many audio files were collected?
"""
llm_out = llm_output(prompt)

prompt = """
%INSTRUCTIONS:
Use the provided pieces of context to answer the query. If you don't know the answer, just say that you don't know, don't try to make up an answer.
%CONTEXT
We stopped the network training after 70 epochs (i.e. a
complete scope of the dataset where each image is used only once), to
prevent overﬁtting. We used a learning rate of 10 −5 , an exponential
learning decay with a Gamma of 0.95, a dropout of 50% and an Adam
Solver type as learning parameters. Those are classic hyper-parameters
for a fast convergence of the network without over-ﬁtting (Srivastava
et al., 2014). The weight initialization is also classic with a random
Gaussian initialization. The training lasted 8 days on our conﬁguration;
we trained and ran our code on a computer with 64GB of RAM, an i7
3.50GHz CPU and a Titan X GPU card for 900,000 images.
%Query
What is the hardware used to execute the code?
"""
llm_out = llm_output(prompt)

prompt ="""
%INSTRUCTIONS:
Use the provided pieces of context to answer the query. If you don't know the answer, just say that you don't know, don't try to make up an answer.
%CONTEXT
If we take a look at the energy consumption from machine learning at the organisational level, Google says that 15 % of the company’s total energy consumption went towards machine learning related computing across research, development, and production [6]. NVIDIA has estimated that 80–90% of machine learning workload is inference processing [7]. Similarly, Amazon Web Services have stated that 90% of the machine learning demand in the cloud is for inference [8]. This is much higher than the estimates put forward by an unnamed large cloud compute provider in a recent OECD report [1]. This provider estimates that between 7–10% of enterprise customers’ total spend on compute infrastructure goes towards AI applications, with 3–4.5% used for training machine learning models and 4–4.5% spent on inference.
%Query
How much energy consumption was used towards machine learning?
"""
llm_out = llm_output(prompt)
