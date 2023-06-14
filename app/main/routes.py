
from flask import Flask, render_template,  request,  Blueprint
from app.models import  Post, BlackandWhite

main= Blueprint('main',__name__)
@main.route("/")
@main.route("/home")
def home():
    page= request.args.get('page', 1, type=int)
    posts= Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)
    blackandwhite=BlackandWhite.query

    return render_template('home.html', posts=posts, blackandwhite=blackandwhite)


@main.route("/layout")
def layout():
    blackandwhite=BlackandWhite.query
    return render_template('layout.html', blackandwhite=blackandwhite)




@main.route("/about")
def about():
    blackandwhite=BlackandWhite.query
    return render_template('about.html', title="about", blackandwhite=blackandwhite)

