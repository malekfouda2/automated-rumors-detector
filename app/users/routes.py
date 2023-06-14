
from flask import Flask, render_template, url_for, flash, redirect, request, Blueprint
from app.users.forms import RegistrationForm, LoginForm, UpdateAccountForm
from app import db, bcrypt
from app.models import User, Post, BlackandWhite
from flask_login import login_user, current_user, logout_user, login_required
from app.users.utils import save_picture









users= Blueprint('users', __name__)
@users.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form=RegistrationForm()
    blackandwhite=BlackandWhite.query

    if form.validate_on_submit():
        hashed_password=bcrypt.generate_password_hash(form.Password.data).decode('utf-8')
        user= User(username=form.username.data, email=form.email.data,password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created for {form.username.data} !', 'success')
        return redirect(url_for('users.login'))
    
    return render_template('regitser.html', title='Register', form=form, blackandwhite=blackandwhite)

@users.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form= LoginForm()
    blackandwhite=BlackandWhite.query
    if form.validate_on_submit():
       user= User.query.filter_by(email=form.email.data).first()
       if user and bcrypt.check_password_hash(user.password, form.Password.data):
            login_user(user, remember=form.remember.data)
            next_page=request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('main.home'))
       else:
            flash('Login Unsuccessfull, Please check Email and password', 'danger')

    return render_template('login.html', title='Login', form=form, blackandwhite=blackandwhite)

@users.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('main.home'))

@users.route("/account", methods=['GET','POST'])
@login_required
def account():
    form = UpdateAccountForm()
    blackandwhite=BlackandWhite.query

    if form.validate_on_submit():
        if form.picture.data:
           picture_file = save_picture(form.picture.data)
           current_user.image_file = picture_file
        current_user.username= form.username.data
        current_user.email=form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('users.account'))
    elif request.method== 'GET':
        form.username.data= current_user.username
        form.email.data= current_user.email
    

    image_file= url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account', image_file=image_file, form= form, blackandwhite=blackandwhite)

@users.route("/user/<string:username>")
def user_posts(username):
    page= request.args.get('page', 1, type=int)
    user= User.query.filter_by(username=username).first_or_404()
    posts= Post.query.filter_by(author_post=user)\
        .order_by(Post.date_posted.desc())\
        .paginate(page=page, per_page=5)
    blackandwhite=BlackandWhite.query
    return render_template('user_posts.html', posts=posts, user=user, blackandwhite=blackandwhite)
