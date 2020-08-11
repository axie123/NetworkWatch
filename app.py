from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///posts.db'
db = SQLAlchemy(app)

# The code that lays out the database for my UI.
class ModelLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    model_type = db.Column(db.String(20), nullable=False)
    description = db.Column(db.Text, nullable=False)
    creator = db.Column(db.String(20), nullable=False, default='N/A')
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return 'Blog post ' + str(self.id)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/posts', methods=['GET', 'POST'])
def posts():
    if request.method == 'POST':
        post_model_name = request.form['model_name']
        post_model_type = request.form['model_type']
        post_description = request.form['description']
        post_creator = request.form['creator']
        new_des = ModelLog(model_name=post_model_name, model_type=post_model_type, description=post_description, creator=post_creator)
        db.session.add(new_des)
        db.session.commit()
        return redirect('/posts')
    else:
        all_posts = ModelLog.query.order_by(ModelLog.date_posted).all()
        return render_template('posts.html', posts=all_posts)

@app.route('/posts/delete/<int:id>')
def delete(id):
    post = ModelLog.query.get_or_404(id)
    db.session.delete(post)
    db.session.commit()
    return redirect('/posts')

@app.route('/posts/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    post = ModelLog.query.get_or_404(id)
    if request.method == 'POST':
        post.model_name = request.form['model_name']
        post.model_type = request.form['model_type']
        post.creator = request.form['creator']
        post.description = request.form['description']
        db.session.commit()
        return redirect('/posts')
    else:
        return render_template('edit.html', post=post)

@app.route('/posts/new', methods=['GET', 'POST'])
def new_post():
    if request.method == 'POST':
        post_model_name = request.form['model_name']
        post_model_type = request.form['model_type']
        post_creator = request.form['creator']
        post_description = request.form['description']
        new_post = ModelLog(model_name=post_model_name, model_type=model_type, description=post_description, creator=post_creator)
        db.session.add(new_post)
        db.session.commit()
        return redirect('/posts')
    else:
        return render_template('new_post.html')


if __name__ == "__main__":
    app.run(debug=True)