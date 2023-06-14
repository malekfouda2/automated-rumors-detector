
import os
from flask import Flask, render_template, url_for, flash, redirect, request, abort, jsonify, Blueprint
import numpy as np
from app.posts.forms import PostForm
from app import  db
from app.models import  Post, Classification, Source, CredibleSources, BlackandWhite
from flask_login import current_user,  login_required
from werkzeug.utils import secure_filename
import joblib
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from fuzzywuzzy import fuzz
import torch.nn.functional as F
from fuzzywuzzy import process
import snscrape.modules.twitter as sntwitter
import requests
import json
import torch
from app.posts.utils import preprocess_input, check_internet_connection
import logging; logging.basicConfig(level = logging.DEBUG)
import pickle
from transformers import BertTokenizerFast,BertForSequenceClassification
import torch.nn as nn
import tweepy

posts=Blueprint('posts',__name__)

@posts.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form=PostForm()
    blackandwhite=BlackandWhite.query
    #clf=joblib.load("akheeer.pkl")

    filepath = os.path.abspath('bertt.pkl')
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    if form.validate_on_submit():

        postContent= form.content.data
        


     




        #postContent=request.json['postContent']
        # input_ids, attention_mask= preprocess_input(postContent)
        # with torch.no_grad():
        #     clf.eval()
        #     outputs= clf(input_ids,attention_mask=attention_mask)
        #     logits=outputs[0]
        #     probabilities= torch.softmax(logits, dim=1)
        #     predicted_label=torch.argmax(probabilities, dim=1)
        #     predicted_label= predicted_label.item()
            
        # preds= jsonify({'label': predicted_label})


        input_text = [postContent]
        MAX_LENGTH = 15
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        classifier = nn.Linear(model.config.hidden_size, 2)

        # Set the model to evaluation mode
        model.eval()

        inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')

        # Pass the input through the model and the classification layer
        with torch.no_grad():
            outputs = model(**inputs)
            logits = classifier(outputs.pooler_output)

        # Get the predicted class
        predicted_class = torch.argmax(logits).item()

        # Map the predicted class index to the corresponding label
        prediction = 'Rumor' if predicted_class == 1 else 'Non-Rumor'

        # def predict_rumor(input_text):
        #     # Load tokenizer and model
        #     tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

        #     # Tokenize input text
        #     input_ids = tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors='pt')['input_ids']
        #     attention_mask = tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors='pt')['attention_mask']

        #     # Make predictions
        #     with torch.no_grad():
        #         outputs = model(input_ids, attention_mask=attention_mask)
        #         logits = outputs[0]
        #         predictions = F.softmax(logits, dim=1).tolist()[0]

        #     return predictions

        # x=predict_rumor(input)
        # if x[0] > x[1]:
        #     prediction='Non Rumor'
        # else:
        #     prediction='Rumor'



        t=[]
        limit = 1


        des=[]
        src=[]
        publish=[]
        urll=[]

       
        if check_internet_connection():
            max_results = 10
            url = (f"https://newsapi.org/v2/everything?q={postContent}&sortBy=relevancy&pageSize={max_results}&apiKey=8bf358a5572d49e2a5713e84661e06d6")
            response = requests.get(url)
            response_json_string = json.dumps(response.json())
            response_dict = json.loads(response_json_string)
            response_dict
            for i in response_dict['articles']:

                    # print('description: '+i['description']+'\n')
                    # print('source: '+i['source']['name']+'\n')
                    # print('publishedAt: '+i['publishedAt']+'\n')
                    # print('url: '+i['url']+'\n')
                des.append(i['title'])
                src.append(i['source']['name'])
                publish.append(i['publishedAt'])
                urll.append(i['url'])
            res = len(des)
        else:
            print('')

            

        
        
        s1 = postContent
        s2 = des
        countR=0
        countN=0
        neutral=0
        percentR=0
        percentN=0
        percentNEU=0
        eval=''
        try:
            for i in range(res):
                if fuzz.WRatio(s1, s2[i])>80:
                    countN+=1
                elif fuzz.WRatio(s1, s2[i]) in range(45,80):
                    neutral+=1
                elif fuzz.WRatio(s1, s2[i])<45:
                    countR+=1
        except Exception as e:
            print('')
                            # print(countR)
        # print(countN)
        if countR > countN:
            eval='Rumor'
        elif countN > countR:
            eval='Not Rumor'
        else:
            eval='Neutral'
        
        userName=''
        isVerified=''
        followers=''

        
        
        # Your code to calculate percentR and percentN goes here
        try:
            percentR=int((countR/res)*100)
            percentN=int((countN/res)*100)
            percentNEU=int((neutral/res)*100)
        except Exception as e:
            print('')
    
        # Handle the exception here
        

        if check_internet_connection():
            access_token = "3251395693-EdEU7h9kbXD5suK68fws3P6HCLJHCl7xmCHJKAb" 
            access_token_secret = "EYKCF6YQbeIx4MXb827XY68pghZRgkT4J2HbHpqSyDoHo" 
            consumer_key = "BNwml9R6rQn48RMAZaWhFu8a4"
            consumer_secret = "9yY24HZQM6XFUf7575qBHuQYPAMRBNZZVcqa4He3iaYAlityhN"

            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            api = tweepy.API(auth)

            tweets = api.search_tweets(q=postContent, count=1, result_type='recent', tweet_mode='extended')
            if tweets:
                tweet = tweets[0]
                userName = tweet.user.screen_name
                isVerified = tweet.user.verified
                followers = tweet.user.followers_count
                t.append(userName)
                t.append(isVerified)
                t.append(followers)
            flash('Check your results!', 'success')
            
        else:
            flash("Check your connection",'danger')   
            
                
        if percentR> percentN :
            black= BlackandWhite(Black=userName)
            db.session.add(black)
        else:
            white=BlackandWhite(White=userName)
            db.session.add(white)

        result= Classification(classifierResult= prediction)
        source= Source( sourceUsername=userName, sourceVerified=isVerified, sourceFollowers=followers)
        credible=CredibleSources(percentr=percentR, percentn=percentN)
        post = Post( content= form.content.data, author_post= current_user, classif=result, sourc=source, cred=credible)

        db.session.add(result)
        db.session.add(source)
        db.session.add(credible)
        db.session.commit()
        db.session.add(post)
        try:
            if res==0:
                percentR=100
        except Exception as e:
            print('')
        return render_template('create_post.html', title='New Post', form= form, prediction=prediction, output2= postContent, one=userName,two=isVerified,three=followers,evalu=prediction,src= '' if not src else src,urll= '' if not urll else urll, blackandwhite=blackandwhite)
    else:
        prediction=''
        postContent=''
        return render_template('create_post.html', title='New Post', form= form, prediction=prediction, output2= postContent, one='',two='',three='', blackandwhite=blackandwhite) 

@posts.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    blackandwhite=BlackandWhite.query
    return render_template('post.html', title=post.content, post=post, blackandwhite=blackandwhite)



@posts.route("/post/<int:post_id>/update", methods=['GET', 'POST'])
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.content = form.content.data
        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('posts.post', post_id=post.id))
    elif request.method == 'GET':
        form.content.data = post.content
    return render_template('create_post.html', title='Update Post',
                           form=form, legend='Update Post')

@posts.route("/post/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('main.home'))