from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired


class PostForm(FlaskForm):
     content = TextAreaField('Tweet', validators=[DataRequired()])
     submit= SubmitField('Detect')
     