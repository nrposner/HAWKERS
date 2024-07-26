Pros and cons of various methods

Word Mover Distance was tried, but is overly literal when paired with our short phrases. Might circle back to it if we have a larger tagged corpus, but until then it's not effective. In particular, a lot of short, positive reviews get antiselected, even having infinite distance.

LSI seems like our best bet still, but I don't understand how similarity interacts with topic count. Need more robust queries


Maybe manually select some reviews which are clearly from one category, and use those as queries? Can synthesize scores from similarities to many of these queries, instead of just one short phrase.
That sounds like it could work, but it could well be a lot of work

Should ask how much data is available overall, outside of what is already available here and translated. Next step would be auto-translating that larger corpus and using it to train a more robust model which we actually put into production


Speaking of, at what stage should I actually implement train/test? Given that it's unsupervised, at no point