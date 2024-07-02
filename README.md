# HAWKERS
This repo contains code developed in June 2024 for Hawkers Madrid.
(Last updated: 7/2/24)

Mission: Use Natural Language processing (NLP) to assess sentiment analysis and classify topics for customer feedback. 

Data structure:
Haven't seen it yet

Expected to be semi-structured, user text submitted via forms

To do (See Project workflow):
Figure out who would be benefiting from this, maybe talk to the customer support team and aks what kind of issues they face
Access data, link to Git
Determine data structure and which tools to use

Tools:
NLTK + TextBlob?

Gensim for topic modeling will probably be our workhorse
Note: Gensim is an unsupervised system

Possibly polyglot? If we wind up not wanting to translate
Need to check how mnay languages we're looking at

We can use SKlearn for some sentiment analysis, but it's not super high level in this field

Data is spread around three Selfoss clouds, in Service, Marketing, and Commerce

Security:
This type of data isn't expected to be sensitive or contain private user data. In theory, it could be made part of a public dataset. However, we need to check and make sure this is the case

Day to day:
Check in w/ Carlos, share this repo with him and Domingo

Style:
Consider implementing black and PEP8
