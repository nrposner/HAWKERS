Data Structure:

Column, integer
"DT_CREATED", date  figure out exact data type
"ID_SERVICE_REVIEW", 24-character alphanumeric code, likely UUID
"CD_LANGUAGE", 2-letter message language indicator, eg "es", "el", etc
    es: Spanish
    el: greek?
    pt: portuguese
    it: italian
    nd: dutch
    fr: french
    en: 
    Total number unknown
"ID_CUSTOMER", numeric customer id, -1 appears to indicate unknown or anonymous
"DS_TITLE", title of the feedback message, often significant overlap with text but sometimes non'overlapping. May be higher density?
"DS_TEXT", message text. Often contains line breaks which makes it difficult to read, can get quite long
"NM_STARS_SCORE", number of stars in the review, from 1 to 5, with more stars indicating higher quality
"DS_TEXT_TRANSLATED": DS_TEXT translated ito English. Appears to be present even if the text was originally in english 