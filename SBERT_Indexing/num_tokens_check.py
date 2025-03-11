from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
tokenizer = model.tokenizer

text = """if you don’t see me then I am traveling  📍 Soma Bay This year: Vienna, Cambridge, Las Vegas, Arizona, New York
Hey everyone👋🏻 sorry for not posting the last days but I am with my family and I wanted to enjoy time and take a little break! But now I am ready to post again and that’s good. 💗
•
*Werbung/Anzeige
@saal_digital gave me the opportunity to create my own photo album and I wanted to save some memories with my brother. I am so happy with the quality and that you can choose between so many designs, materials and sizes for your book! So if you’re still looking for some valentines presents this book is a really good idea because every book is individual! Thanks to @saal_digital!
Have a great evening and see you tomorrow! 💋
•
•
•
•
•
#germanblogger #saaldigital #promotion #thehague #instadaily #lifestyle #travel #lifestyleblogger #photography #happyme #girl #instagood #dailypost #february 
Hey everyone👋🏻 I spent my day in Frankfurt and enjoyed shopping😉 my favourite street is definitely the Goethestreet. Every shop is special and beautiful! I was freezing a lot but who wants to be beautiful must suffer! We took a @smart_deutschland from @car2go and it was a very cool experience because I seriously have never been driving with a smart😂 what have you been doing today? Have a nice evening💋
∙
∙
∙
∙
∙
#americanstyle #germanblogger #citytrip #travelstyle #frankfurt #coffeefellows #lifestyleblogger #instadaily #fashion #inspiration #lifestyle #instagood #prettylittleiiinspo #hairsandstyles 
Waiting for my first @kenzo fashion show like...😂 I want to be with someone who is doing everything in life, but nothing on a rainy sunday afternoon. @atticuspoetry"""

tokens = tokenizer.tokenize(text)
print("Number of tokens:", len(tokens))
