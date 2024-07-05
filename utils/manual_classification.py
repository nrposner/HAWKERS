import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models, similarities
from collections import defaultdict


data = pd.read_csv("data/service_reviews_15000rows_translated.csv")

# product quality queries

pq_queries = [
    'I love the quality of the glasses!',
    'Good value for money. Good designs!',
    'Love the style and the high quality of the sun glasses',
    'I didnt liked the quality of sunglasses',
    'A particular model and good quality of the lens, this is the second time I have purchased this brand',
    'Everything great as always. I have been a user for years and it never disappoints.',
    'Great product',
    'Incredible quality and price, in love with this brand!!!!!!',
    'Quality glasses. I have more than 10. Very interesting price. Thank you so much',
    'The quality of the glasses is a bit low. It is the second time I have bought and the truth is that the quality of these models was not up to par.',
    'Great, fresh designs, unisex. That looks good on any type of face',
    'Good product at a good price.',
    "I have already bought several glasses, it is already a classic every year. Nice design, good quality and good price. I don't give 5 stars for the glasses box, it should be a little better.",
    'A great offer, good product in every way and it is not the first time I have bought.',
    'Poor quality',
    'Good quality glasses - price. I got a 2x1 offer, I already had some and I bought them again, happy with them.',
    'Excellent quality-price ratio, really beautiful.',
    'Everything OK',
    'Designer glasses of good quality, they meet expectations.',
    'Perfect and useful.',
    'Style, simplicity and modern',
    'Beautiful sunglasses, the best quality and price ratio. I love my hawkers.',
    'Everything very good, fast and reliable shipping.\n\nI will repeat.',
    'Quality glasses at an exceptional price. I will purchase them again.',
    'Very nice sunglasses! Decent price!',
    "It's what I expected, good product/price ratio",
    'I was very satisfied with my purchase for its price, quality and speed of shipping, thank you very much',
    'Plastic frame but very good lenses.\n\nIdeal for summer\n\nExcellent value for money',
    'I love the quality of their products. This is my second pair and I love them.',
    'Just as shown, very pretty and sturdy.',
    'I love the brand and especially the 3x1 promotions',
    'I will continue to buy. ⭐⭐⭐⭐⭐',
    'It looks great with them and the designs are very cool, they have a lot of variety to choose from.',
    'Luxury I will buy again', 
    "It was fast. I'm not impressed with the glasses. They cope very poorly with the reflection of light inside the glasses. when i wear them i see reflections inside the lenses. The glasses would need an anti-reflection coating.",
    'Very fast and easy purchase and delivery. Everything was great and the glasses were perfect and discounted.',
    'They are very attractive',
    'I absolutely love them!!!!.....and the offers you have are ideal!!!!....thanks!!!!',
    'There are no sizes',
    'They always comply', 
    'Cool and practical fit',
    'The measures on the site were different from the measures on the glasses. For example, there was written on site 18 mm bridge for fusion clear blue one polarized and on the actual glasses it is written 17',
    'everything very good, thank you very much',
    'Lenses that reflect inside...annoying.', 
    "I had been wanting to buy these glasses for a long time and for one reason or another I didn't. But in the end I decided and I honestly don't regret it, it was an excellent purchase, I bought some for myself and I bought others for my wife and I was also very delighted. Without a doubt we will buy again and above all recommend them.",
    '2nd time purchasing.! Everything went excellent!',
    'Everything was fine'
    'Very fast and everything correct. Good quality of the glasses. It would be nice if it included a cloth to clean them.',
    'Great, super pretty glasses. and good material',
    'I have ordered twice and the 4 glasses have arrived in perfect condition, very good quality and the packaging is very careful and with excellent details. Yes, cleaning wipes would be good!!!',
    'I ordered a pair of sunglasses from their wide selection, I was able to benefit from a special advantage and get 2 pairs for the price of one. I find the quality/price ratio very good and the shipping went well, fast reception.\n\n\n\nI will not hesitate to order again.',
    'Second pair and great value for money',
    'Hawkers are good glasses at a good price. And if we add a good promotion to that, then there you have it. Glasses to enjoy the summer.',
    "I like it because you send me good offers from time to time, and that's when I can buy. I already have a good collection so I'm waiting for another offer like the last one that came out super cheap.",
    'Well done site and product value for money is top notch',
    'A particular model and good quality of the lens, this is the second time I have purchased this brand',
    'Everything great as always. I have been a user for years and it never disappoints.',
    "The worst glasses I have ever bought. Terrible quality, both in lenses and the frame. \n\nI don't recommend them at all",
    'I have liked my glasses that I have purchased so far with you, they look great and look wonderful, thank you',
    'Great sun glasses and comfy to wear',
    'Excellent.. and the ottima quality',
    'Great product',
    'Quality glasses. I have more than 10. Very interesting price. Thank you so much',
    'The whole process was great, from start to finish!',
    'Good product at a good price.',
    'Excellent service and material with products very good price-quality ratio',
    "I couldn't be happier with this collaboration, and with my glasses, I hope you continue counting on me",
    'The glasses are nice. We took 4 and three of them were good since they are for men. But there are glasses that are for women that are very, very large in size. I wish there were smaller sizes',
    'My blue light protection glasses were scratched the first day',
    "Men's glasses are too small",
    'Today I went to the Hawkers store in Córdoba and the glasses are beautiful and the service has been magnificent and 10',
    'From the best glasses companies! Excellent quality!',
    'Nice products',
    'I have already ordered my sixth pair of glasses from them. Always perfect and beautiful. Very comfortable and as gifts they were very well received.',
    'Top glasses', 
    'Quality and good prices',
    'The models are very cute and comfortable. They fit wonderfully AND give a very good result. These items are the fourth Hawkers glasses I own.',
    'Incredible experience and quality glasses. I recommend it to everyone, it is well worth it.',
    'Very good product and service',
    'Excellent products',
    'Perfect, great sunglasses. Good quality/price',
    "Everything is ok, I haven't tried them yet but they are well made. The only flaw is that instead of putting stickers in the package, the cloth to clean the lenses is more useful.",
    'The quality of the glasses is terrible. I bought the last ones 2 years ago, so I can compare. The plastic of the frame and temples are of very low quality.',
    'Phenomenal glasses...quality and good price.',
    'Very light and comfortable',
]

# mixed

mixed_queries = [
    'I ordered a pair of sunglasses from their wide selection, I was able to benefit from a special advantage and get 2 pairs for the price of one. I find the quality/price ratio very good and the shipping went well, fast reception.\n\n\n\nI will not hesitate to order again.',
    'Good glasses, good promotions, good prices and good delivery service.',
    'Everything great quality price and fast shipping, a ten',
    'Speed, quality and design, very happy, thank you and very good price!!!!',
    'Very fast shipping and the product as I expected, it has been a great success',
    'Sunglasses arrived quickly and great for the price. Had already bought some before and the quality was first class so have purchased more.',
    'Good, quality products. Very fast shipping.',
    'Good product, good price, fast shipping. I recommend it',
    'Seriousness, speed in order management, very good quality at a very affordable price.',
    'The lenses themselves are expensive for the quality of material they are made of. In terms of shipping the product, it was quite good',
    'Very good product quality and great service at Hawkers Añaza',
    'really good and fast',
    'I love glasses!!\n\nImpeccable shipping!',
    'Fast shipping, quality products, very happy with everything.',
    'Fast and perfect product',
    'Good quality products and timely attention in the virtual store.',
    'Price little too high :-)', 'fast and everything good',
    'Very fast, good price and quality',
    'Order received quickly and efficiently...price without competition, quality and feel great',
    'I loved the quality of the product. Very good quality and super comfortable sunglasses.\n\nReceived within the correct deadlines. All good',
    'All right. Fast shipping', 
    'Best product and fast shipment',
    "Good price and good service.\n\nThe quality for the price is good, but you have to understand that with these discounts you are not going to buy the same thing as for three times as much.\n\nLots of variety and fast service.\n\nIt's a shame that returns are not more accessible.",
    'Fast, good quality/price',
    'Great deals and fast shipping',
    "The quality is on a good standard, not the best, but at least they have a minimum standard. The price itself is a bit high, tho I can say they almost always have the 2x1 or 50% discount, then the price is worthy. Their assistance is not good. Once they sent me a wrong pair of glasses, I contacted them and they sent me the right one quite soon letting me keep the wrong pair. Another time I had a quality problem with one of their glasses, they didn't ever even reply to my emails and contact requests... 3/5",
    "Awful!! Let no one understand here...\n\nI placed an order, the delivery times were not met and I no longer needed the glasses, I had to buy others, I tried to cancel the order and it is impossible!!! ORDERS CANNOT BE CANCELED!! THEY HAVE NO WAY TO DO IT THROUGH THE WEB...\n\nI wrote dozens of messages to which they did not respond even once, nor did I receive the typical emails saying that the messages were sent and that I was waiting for a response... I opened a dispute on PayPal... thank goodness I paid with them... They didn't answer them either... \n\nI raised the dispute to a claim after waiting a couple of days, and so far they haven't responded either... \n\nOf course, after 3 weeks they send me a message saying that they are preparing my order and that the transport company will deliver it to me in a few days...\n\nI DON'T WANT THE ORDER NOW, I HAVE SAID IT ACTIVELY AND PASSIVELY, THROUGH THEIR PAGE AND THROUGH PAYPAL FOR 2 WEEKS AGO!!!!\n\nNobody answers anything, they don't have a phone to call, and their return policy is also disastrous, they make you pay the costs.\n\nDo not buy on the page, an online service like this cannot currently be provided.",
    'Positives that it took very little time to bring the package, the negative that the glasses came bent',
    "Everything was fast and correct. I'm not giving it 5 stars because they seem a bit dark to me.\n\nThat's what happens when you buy things online without trying them on first!!",
    'You do an impeccable job. Orders arrive quickly and you always respond to emails.',
    'Friendly treatment and very good service. Super cool glasses and great prices',
    "I really like the service and the glasses, the only thing I don't like is the service which lacks the ability to change them for free if the model is not to your liking, as it creates mistrust and uneasiness when ordering glasses without knowing how they will look.",
    "I'm back for another 2 for 1 offer to give a couple of gifts. As always, the clerks advised me perfectly and controlled the measures against covid. Very happy with my purchase!",
    'I placed my order HWEU0098170 on June 20th, I have not received a delivery note number or a response regarding the delivery of my order. I have filled out the contact form on the page 5 times, without receiving a specific response regarding my delivery. I feel scammed! Do not buy from that page.',
    'The glasses are what I expected, very comfortable and adapt to the face perfectly. \n\nThe purchase was very easy and it arrived very quickly. I recommend it',
    'Slow delivery and over priced! Delivery took over a week which in these days is a bit ridiculous for a pair of sunglasses.. I bought one of the "premium" pairs but they feel incredibly cheap and are definitely £35 worth. If I could I would have returned them but with poor customer support and no clear way of how to return them I didn\'t want to risk sending them back and not receiving a refund.',
    'I received my glasses on time. Very good quality, very satisfied! Thank you',
    'Super fast delivery and the glasses are fantastic. I recommend them.',
    'Excellent service and fast delivery',
    'Fast shipping. The glasses are very pretty and weigh very little. Perfect. There are many models.',
    'Very fast delivery and the glasses are really cute I loved it',
    'Super fast, and very good validity',
    'The lenses meet expectations, delivery time was very fast.',
    'Fast and recommended',
    '... top quality.\n\nawesome products, profesionalism and short time for delivery.',
    'Fast delivery, top quality',
    'Very fast delivery and good product',
    'Very fast and easy purchase and delivery. Everything was great and the glasses were perfect and discounted.',
    'Everything is perfect! Fast delivery, great website, good price.',
    'Slow delivery and over priced! Delivery took over a week which in these days is a bit ridiculous for a pair of sunglasses.. I bought one of the "premium" pairs but they feel incredibly cheap and are definitely £35 worth. If I could I would have returned them but with poor customer support and no clear way of how to return them I didn\'t want to risk sending them back and not receiving a refund.',
    'Orders arrive super fast and at incredible prices!',
    'super fast delivery, very good product quality and a great offer',
    'Good value for money and delivery was quite fast.',
    'Super fast delivery and the glasses are fantastic. I recommend them.',
    'Good products and fast delivery!',
    'Great variety at a good price. Happy with the delivery.',
    'Fast delivery, quality, light and comfortable.',
    'The lenses meet expectations, delivery time was very fast.',
    'Fast delivery and quality original glasses',
    'Fast delivery. Quality product.',
    'The delivery was super fast and the product packaging was very cool.',
    'Punctuality at delivery time\n\nThe product is as described on the website',
    "Excellent service and fast delivery. It's not the first time I buy and I love it. I will repeat.",
    'I was amazed by the fast delivery (in Ro)- i had the glasses the next day.\n\nVery good quality and a very good fit - love them',
    'quick delivery nice package good glasses',
    'Beautiful glasses, fast and easy delivery',
    'fast, pretty, and economical',
    'Fast and great. I will repeat for sure',
    'Great delivery service. As always the products are of very good quality and the offers are really cool.',
    'Everything perfect, fast and very good service',
    'Well finished glasses, good polarized lenses, impeccable delivery',
    'I love their glasses Many models and good prices especially when you can apply discounts Very fast delivery',
    'For the offers and fast delivery, without neglecting the quality of their products',
    'The glasses arrived super fast and of very good quality.',
    'Excellent quality at unbeatable prices. The most vfm and stylish glasses on the market. It took only 4 days from order to receipt.',
    "I don't like that they charge shipping costs and if they don't fit you, you have to return them and lose that money. For those of us who don't have a store in our city to try them on, it's a hassle.\n\nOtherwise, there are good offers and very cool models.",
    'I have been a Hawkers customer for a long time, very good value for money, it takes two days at most to arrive and if you have any problems the incident team will solve it as soon as possible.',
    'Good, quality products. Very fast shipping.',
    'Good product, good price, fast shipping. I recommend it',
    'Speed, quality and design, very happy, thank you and very good price!!!!',
    'Very fast shipping and the product as I expected, it has been a great success',
    'Sunglasses arrived quickly and great for the price. Had already bought some before and the quality was first class so have purchased more.',
    'As always, very fast service, they are super friendly and help you right away with any questions you have about the product or the purchase. And the cool, good quality glasses :)',
    'The glasses are great. Quality, the packaging is great and the order arrived quickly! Top',
    'Fast shipping, quality products, very happy with everything.',
    'The order arrives very quickly and the quality of the glasses is good.',
    'Fast shipping and excellent presentation of the products',
    'Excellent products, arrived on time.',
    'Fast shipping. Good and economical sunglasses.',
    'Excellent quality. Very fast delivery.',
    'Everything very good, fast and simple.',
    'The order and delivery was as described. Good service and presentation of the purchased product.',
    'I was bit dubious about ordering reading the reviews. But such a good deal for some stylish sunglasses didnt want to miss out . Ordered them then got regular updates on progress. Arrived the next week exactly what I ordered. Packaged great and look fabulous. \n\nCannot fault and delivered by Hermes in the uk no problem.',
    'Fast, safe and quality',
    'I love the brand!! In addition to the fact that they are quality glasses, super light and comfortable, shipping is very fast and there are always interesting promotions available!!',
    'Excellent product: Price/Quality, fast management.',
    'You are my reference glasses. Lots of quality, design and service.',
    'The delivery time was very fast and the product arrived in good condition, as shown in the publication.',
    'Perfect, everything went 5 stars', 
    'Perfect article.', 'Everything fast and good',
    'Excellent product and very quick and easy',
    'Good product quality price and very fast delivery.',
    'I arrived at the store on Carretas Street and the people who work there welcomed me in an unbeatable way, monitoring all the anti-covid measures, and offering me their promotions. Delighted with my purchase and with the people who work there.',
    'The products came in time and as expected.\n\nThe products are always value for money and the quality is good.',
    'The shipment arrived on time and the packaging was in great condition. No complaints whatsoever. The whole structure of the glasses is a bit fragile. But for the price (buying on offer) and the polarized lenses, I am content with the result',
    
]

# delivery service

ds_queries = [
    "I didn't receive my product but i already have buy for those...",
    'Speed \u200b\u200band effectiveness',
    'Good shipping everything perfect', 
    'I have not received the product yet, I am sorry to give you a bad rating, but your delivery company has not worked correctly, I am correcting my previous opinion because I contacted the company and it was resolved.',
    'Fast shipping. They arrived safely, there were no problems with shipping.',
    "It arrived very late and now I'm trying to return it",
    'The order arrived quickly and correctly, top!',
    "The glasses are amazing but the transportation service is painful!! This is the last time I buy from their website again, I've been waiting for a package for more than a week and I still haven't received it, on top of that if you want to contact them you have to call 902!! Zeleris the worst transportation company, I don't understand how they can work with them!!!",
    'The order arrived quickly and in perfect condition.',
    "Hawkers is great, the delivery company is disastrous, terrible, they don't meet deadlines and talking to them is like talking to a wall.",
    'It came super fast and the delivery man was very friendly, I will definitely repeat.',
    "It said they sent a gift and they didn't send anything. If they hadn't said they sent a gift it would be a 5*",
    'Extremely quick delivery once the order had been processed, which took a little longer than the 24-72hrs advertised.',
    'Was very concerned after reading these reviews and I have been burned by companies before not delivering items! Unfortunately I read these reviews after I ordered the sunglasses! Despite expecting them not to show up, I received several emails telling me where they were and they arrived today! Only took a week which is brilliant from Spain! Cant wait to give them a try! Thank you!',
    'The delivery of the product was very fast, in one day I already had them. In addition, all the glasses were very well protected.',
    'Fast and good delivery!',
    'This company did not send a confirmation email of my order, then my order went "missing" from the delivery service. Would not recommend.',
    'Fast delivery and everything correct.',
    'Excellent service and fast delivery',
    'Fast delivery', 
    'Top. Fast and professional',
    'Delivery on time and everything is fine',
    'Top quick delivery',
    'Perfect delivery, arrived earlier than expected. The merchandise in perfect condition.',
    'I made a purchase and was not aware of anything between purchase and shipping, however the specified delivery days have passed and no order has arrived!',
    "I'm more than 5 days past the scheduled delivery day of the order and I don't have any response. Very bad experience",
    "I paid 9 for fast delivery, still nothing 1 week after the original delivery day.\n\nUPS still doesn't know where to deliver!",
    'The delivery has been very fast',
    'Everything perfect, phenomenal treatment and everything passed quickly.',
    'Quick delivery',
    'Easy to go through the entire choosing and ordering process. Delivery was also quick and within the stated time frame.',
    'EVERYTHING PERFECT, VERY GOOD WEBSITE AND FAST DELIVERY, A 10',
    'Unfotunately I have not received the delivery.',
    'Delivery time in good time',
    'Great and fast treatment',
    'Terrible shipping service. The courier threw the package into the backyard',
    'It was a great experience in the speed and Genius service',
    'I love glasses!!\n\nImpeccable shipping!',
    'Good shipping everything perfect',
    'I have ordered 4 pairs from them with no problems each time and there all great. However on the 5th pair that I ordered on the 3rd of July I have not received the shipment email it could be a back log as there was a big sale on at the time of the order  I will wait a few days before I take it further as they gave been great 4 times in the past (1 within the last 30 days) \n\n\n\nI will update this when I know more.   \n\n\n\n*Update*\n\n Received  notice on here and via email on the 10th that they will be shipped and they \n\n turned up on the 11th\n\n\n\n\n\nI have  good service from hawkers  over the years  and will keep going back. I can only assume that they have been really busy during the sale \n\n\n\nBut overall i am super happy with the glasses/hawkers',
    'My order has been lost for a week. It appears to me as delivered and it has not arrived and no one gives me a solution',
    'I did not receive my order for Aanuel glasses',
    "The product as such, perfect. I chose 4 glasses using a very good promotion and I loved them all. \n\nNow, the shipping service is zero! They decided, without my authorization, to leave my package with a neighbor (without knowing if that was going well for me, if I get along with said neighbor...) when I was able to talk to them (who were already leaving the package to the neighbor in question) I told them that I preferred that they leave it at their delegation and that I would pick it up, my surprise was when the only delegation is in a different town, without transportation (I don't have one) it was impossible for me to get there so I had to resign myself. \n\nThey should change the courier company for one that has several branches (I'm talking about Bcn city, not a remote town) or use the delivery service at collection points, as the vast majority of companies already do today and which It's great for those of us who buy online but don't usually spend much time at home.",
    "I still haven't received confirmation of the shipment, I sent an email to support and they didn't answer me either, very bad experience",
    'You have had many problems delivering the package.',
    "I placed an order on 4/27/2022 to give as a gift for Mother's Day on 5/1/22, it is 5/4/22 and I still have not received the order.",
    "I still haven't been delivered!!!!! \n\nI filed a complaint because the delivery man told me that the package was in my box!!!!! \n\nBut I didn't receive anything",
    'Speed \u200b\u200bin product delivery.',
    'Although the shipping information was scarce, it was so fast, I almost preferred it.',
    "The lenses are very good, in fact I have bought several over the years. What I didn't like on this occasion was the logistics, I bought them 10 days ago, the package has been in the city for 5 days and they had not moved until yesterday (2 days after the promised date) when they tried to deliver them , the only day that there was no one to receive the package. Now I don't know until when they will try to deliver it again (because apparently they don't deliver on weekends), the parcel service has not contacted me again nor did they tell me if there was any other way to pick them up.",
    "The address was incorrectly copied to the package, it is returning to sender. Although i ordered faster shipping, I'm still waiting for the package.\n\n\n\nUpdate 18.8.2022 -  finally received my package (ordered on 17.7.2022). Somehow they managed to copy the address incorrectly even the 2nd time. The glasses.\n\n\n\nThe aviators look fine, however the wayfarers squeek and look like inferior quality",
    'They have always delivered and have never let me down.',
    'Hello, I placed an order on November 15, 2019. The glasses have not been delivered to date. Zero contact, automatic e-mail. I lost 52 on the trade.',
    'Fast and secure',
    
    
]

# customer service

cs_queries = [
    "They sent me some Northweek glasses with defects, I informed them of the incident and told them to study my claim. After 10 days I'm still waiting for a solution. \n\nAfter a few days, they sent me new glasses at no cost. In the end, customer service did its job and well done. Thank you",
    'Very efficient staff attention.', 
    'I had an order of 7 pairs of sunglasses while when my order was delivered, there were only 5 inside. I have been writing to customer service more than three times already during the last three weeks and no reply so far. The company definitely has to better their process of dealing with the orders and especially customer service if there is any problem after they deliver it...... two pairs are a lot and so far i have NOT gotten my sunglasses.....',
    'The customer service is very bad as well as the after-sales service.',
    'There was a missing item from the order and when I tried to contact to arrange delivery there has been zero response',
    "I didn't have any problems in the past, ordered 2 times, delivery was always on point. But now I ordered 3 pairs on 27th of June...still no information at all. Could you please provide me with information because I need these glasses till 20th of July.\n\n\n\nORDER : HAWKERS2280050",
    "Since July 22, when I started my return, I'm still waiting for the package of two glasses that I ordered to be picked up, because they don't fit me well.\n\nI filled out the form several times and even tried to return them in the store, but they told me that they had to contact me because the return must also be online.\n\nAnd here I'm still waiting.........",
    "I placed an order for 2 Carbon Black Dark One LS sunglasses 2 months ago and I still haven't heard from my glasses.\n\nI have sent several claim forms and no attention has been paid to me.\n\nTo collect they charge quickly but to deliver the order no matter, and you try to contact them via contact through their website and nothing\n\n\n\nTerrible service and feeling of total scam",
    'There was a missing item from the order and when I tried to contact to arrange delivery there has been zero response',
    'Good evening, I received the order and one glass was missing, I have sent through the site and no one has answered me.',
    "I was a Hawkers customer and thought it was a trustworthy brand, but it wasn't. \n\nOn November 19th I placed two repeat orders (my mistake) and then tried to cancel one of them (unsuccessfully). I sent an email and they told me not to accept the order at home as it would be returned (and it was). \n\nTo date, I have sent 3 emails demanding a refund and still nothing! They have the glasses and my money (65).\n\nUntil reasons to the contrary, I do not recommend this brand TO ANYONE!!!\n\nThere's no point in saying to report the problem at the link because I've already done it!\n\n\n\nUpdate: I haven't received my refund yet. I received an email saying that they had refunded me but I still haven't received the money.",
    "I have received the box open and crushed, and the plastic envelope open, the glasses with the broken temple and very poor quality.\n\nThere is no contact telephone number, and when you fill out what they ask for the return, they ask for information that does not appear. They also did not send me an email about the order with any information.\n\nThis is the first time this has happened to me, I won't buy again.\n\nThat after a year they respond to me by directing me to the same page where contact is not possible, seems shameful to me.",
    "I had a problem with one of the glasses, I've already sent several contacts and so far I haven't received any response!!\n\n\n\n06/28 - Finally, I got a response and the situation is resolved.",
    'This afternoon I was spending some time with my family at the Salera shopping center (Castellón de la Plana) and passing by this store my 21-MONTH-OLD baby started screaming and crying because he was\n\ntired. The shop assistant at such an optician has started yelling at us and telling us to shut up the "f***ing child" who has a headache from crying so much, she has told us\n\nstarted insulting and telling us to get out of there when we were just passing by\n\nof the store WE HAVE NOT EVEN ENTER THEIR PREMISES. She told us to tell the baby 21\n\nMONTHS for him to be silent as if the child would understand. IT IS A SHAME THAT YOU HAVE THIS TYPE OF CRAP WORKING IN FACE OF THE\n\nPUBLIC, IF YOU\'RE NOT ABLE TO EMPATHIZE WITH A BABY, I CAN\'T IMAGINE HOW YOU\'RE GOING TO TREAT A BABY.\n\nTHE CLIENTS. That person should be in a psychiatrist or rather a zoo. All this has\n\npast now at 19:43 (05/04/23).\n\n\n\nCLARIFICATION: I have always liked this business in terms of quality and customer service, but with this experience I VERY DOUBT that I will enter your store again.',
    "Order placed on 02/05/2021.\n\nToday 22/06/2021 I still haven't received anything. I go to the post office to track it and they can't find out where this package ended up (tracking number is not there).\n\nI sent dozens of emails as advised by you (Hawkers), I reported it on the Facebook page.\n\nNo response.\n\nIt's inconceivable.\n\nI will file a complaint with the Guardia di Finanza, because this is a scam.\n\nAnd I will ask for compensation from your company through a lawyer.\n\nI'm writing to you again even if it's absolutely useless:\n\n\n\nOrder: HAWKERS3493644\n\nTYPQA5L00796956\n\n\n\nAnd don't give me the usual stupid answers under the post, contact this one and contact that one. I've already done everything that had to be done.\n\nSolve the problem or refund me because the crime of fraud and theft is triggered.",
    'Terrible, do not buy glasses with this company here in Colombia, they do not respond to emails or social networks and they do not even apologize for not responding or anything, the customer service is terrible.',
    'They always respond to provide a solution to any problem',
    "I sent three emails because the glasses were defective and I still didn't get a response and it's been a week now, embarrassing.\n\n\n\nYou told me to contact you but where? Why don't you read the emails, give me another solution, do you have a phone number?",
    'Terrible!\n\nI made an order 2 weeks ago and have not received my order. Ive emailed multiple times and have received no responses. It must be a SCAM!?',
    "The after-sales service is terrible, I have been trying to get a response for more than a week for a shipment that was sent wrong and no one answers me or gives me a solution\n\nEDIT: They answer me here that they have not received any official message when I have used the link on their page on four occasions and also to the email they sent with the order. I HAVE GOT NO RESPONSE IT'S BEEN TWO WEEKS, a shame!",
    
    
    
]

#nonspecific praise, maybe file unde rproduct quality?

praise_queries = [
    'Fantastic!',
    'All very well',
    'All perfect',
    'Always fantastic',
    'Very good experience',
    'Very satisfied',
    'All\n\nExcellent',
    'Excelent as always',
    'Great...',
    'Everything perfect',
    'Everything is great.',
    'I love them!',
    'Love everything about Hawkers!',
]

# other

other = [
    "yes I don't speak spanish",
    
]


cs_queries = list(set(cs_queries))
ds_queries = list(set(ds_queries))
pq_queries = list(set(pq_queries))
mixed_queries = list(set(mixed_queries))

all_queries = cs_queries + ds_queries + pq_queries + mixed_queries

train_list = list(data.iloc[:, 8].values)

train_corpus = pd.Series([x for x in train_list if x not in all_queries])

query_dict = {"Customer Service": cs_queries, "Delivery Service": ds_queries, "Product Quality": pq_queries, "Mixed": mixed_queries}




#revised version

def preprocess(corpus:pd.core.series.Series, min_len:int = 3, max_len:int = 15) -> list:
    """ Take in a corpus of text in a pandas series and perform
    preprocessing

    corpus: a pandas series containing text

    min_len: minimum word length. No shorter words will be retained

    max_len: maximum word length. No longer words will be retained
    """
    
    if not (min_len <= max_len):
        raise ValueError("make sure your minimum and maximum token lengths are not reversed")

    preprocessed_corpus = []

    for i in corpus:
        preprocessed_doc = gensim.utils.simple_preprocess(i, min_len = min_len, max_len = max_len)
    
        preprocessed_corpus.append(preprocessed_doc)

        # go line by line, removing common words
    stoplist = set('for a of the and to in'.split(' '))
    texts = [[word for word in document if word not in stoplist]
         for document in preprocessed_corpus]

    # count word frequencies
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

    return processed_corpus

def corpus_maker(processed_corpus:list):
    """ Take in a processed corpus from preprocessing and transform it into
    a tfidf bag of words corpus
    
    """
    # turn this into a dictionary structure
    dictionary = corpora.Dictionary(processed_corpus)

    # create a 'bag of words' corpus using that dictionary
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # train the model

    # tfidf is a transformation that finds term frequency in model frequency
    # we will use this in order to create a structure which other models can attack more easily
    tfidf = models.TfidfModel(bow_corpus)

    corpus_tfidf = tfidf[bow_corpus]

    return corpus_tfidf, dictionary

def mean_similarity(corpus_tfidf: gensim.interfaces.TransformedCorpus, dictionary: gensim.corpora.dictionary.Dictionary, queries:list[str], mod, num_topics:int):
    """ Take in a tfidf corpus and dictionary created by corpus_creation,
    as well as a query such as 'customer support' and a model. We generate similarity scores 
    for each document across all queries, and take the mean of all scores in each category of query. 

    This remains a naive classifier, more refinement is needed. LSI model is recommended

    mod: must be formatted as models.ModelName, such as models.LdaModel, or models.LsiModel
    """

    model = mod(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    
    
    
    df = pd.DataFrame(columns = queries)
    
    
    #want to output a single number for each corpus element, the mean
    #score across all queries
    
    for q in queries:
        vec_bow = dictionary.doc2bow(q.lower().split())
        vec_model = model[vec_bow]  # convert the query to LSI space

        #index these
        index = similarities.MatrixSimilarity(model[corpus_tfidf])

        sims = index[vec_model]  # perform a similarity query against the corpus
        
        df[q] = sims

        
    scores = df.mean(axis=1)
    
    return scores
    


def classification_pipeline_2(train_corpus, query_dict, mod, num_topics):
    
    #query sets should bea dictionary of lists
    
    processed_corpus = preprocess(train_corpus)
    corpus_tfidf, dictionary = corpus_maker(processed_corpus)
    df = pd.DataFrame(columns = list(query_dict.keys())) ##check
    
    
    for item in query_dict.items():
        scores = mean_similarity(corpus_tfidf, dictionary, item[1], mod, num_topics)
        df[item[0]] = scores
    
    
    df["class"] = df.idxmax(axis=1)
    df["text"] = train_corpus


    bins = []
    
    for q in list(query_dict.keys()):
        subset = df[df["class"]==q]
        bins.append(subset["text"].values)

    classes_dict = {}
    for q, b in zip(query_dict.keys(), bins):
        key, value = q, b
        classes_dict[key] = value
        
    return classes_dict
