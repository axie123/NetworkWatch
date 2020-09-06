from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

# Loading the evaluation data.
content = open('eval_data.txt', 'r')
content = content.read().splitlines()
model_eval = json.loads(content[0]) 

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///models.db'
db = SQLAlchemy(app)

# The code that lays out the database for my UI.
class ModelLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    model_type = db.Column(db.String(20), nullable=False)
    description = db.Column(db.Text, nullable=False, default='No Description')
    creator = db.Column(db.String(20), nullable=False, default='N/A')
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    train_ratio = db.Column(db.Float, nullable=False)
    test_ratio = db.Column(db.Float, nullable=False)
    train_batch_size = db.Column(db.Integer, nullable=False)
    test_batch_size = db.Column(db.Integer, nullable=False)
    epochs = db.Column(db.Integer, nullable=False)
    lr = db.Column(db.Float, nullable=False)
    mom = db.Column(db.Float, nullable=False)
    reg = db.Column(db.Float, nullable=False)

    loss_func = db.Column(db.String(20), nullable=False)
    opt = db.Column(db.String(20), nullable=False)
    rdm_seed = db.Column(db.Integer, nullable=False)
    log = db.Column(db.Integer, nullable=False)
    opt_args = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return 'Model number ' + str(self.id)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models', methods=['GET', 'POST'])
def models():
    if request.method == 'POST':
        model_name = request.form['model_name']
        model_type = request.form['model_type']
        description = request.form['description']
        creator = request.form['creator']

        train_ratio = request.form['train_ratio']
        test_ratio = request.form['test_ratio']
        train_batch_size = request.form['train_batch_size']
        test_batch_size = request.form['test_batch_size']
        epochs = request.form['epochs']
        lr = request.form['lr']
        mom = request.form['mom']
        reg = request.form['reg']

        loss_func = request.form['loss_func']
        opt = request.form['opt']
        rdm_seed = request.form['rdm_seed']
        log = request.form['log']
        opt_args = request.form['opt_args']

        new_des = ModelLog(model_name=model_name, model_type=model_type, description=description, creator=creator, 
        train_ratio=train_ratio, test_ratio=test_ratio, train_batch_size=train_batch_size, test_batch_size=test_batch_size, epochs=epochs,
        lr=lr, mom=mom, reg=reg, loss_func=loss_func, opt=opt, rdm_seed=rdm_seed, log=log, opt_args=opt_args)
        db.session.add(new_des)
        db.session.commit()
        return redirect('/models')
    else:
        all_models = ModelLog.query.order_by(ModelLog.date_posted).all()
        return render_template('models.html', models=all_models)

@app.route('/models/view/<int:id>', methods=['GET'])
def view(id):
    if request.method == 'GET' and model_eval['id'] == id:
        model = ModelLog.query.get_or_404(id)
        data = {'training_interval':model_eval['training_interval'],'training_loss': model_eval['training_loss']}
        return render_template('view_model.html', model=model, data=data)
    else: 
        return render_template('error.html')

@app.route('/models/view/<int:id>/valid_loss', methods=['GET'])
def view_valid_loss(id):
    if request.method == 'GET' and model_eval['id'] == id:
        model = ModelLog.query.get_or_404(id)
        data = {'valid_interval':model_eval['validation_interval'],'valid_loss': model_eval['validation_loss']}
        return render_template('view_model_validl.html', model=model, data=data)

@app.route('/models/view/<int:id>/testing_loss', methods=['GET'])
def view_testing_loss(id):
    if request.method == 'GET' and model_eval['id'] == id:
        model = ModelLog.query.get_or_404(id)
        data = {'test_interval':model_eval['testing_interval'],'test_loss': model_eval['testing_loss']}
        return render_template('view_model_testl.html', model=model, data=data)

@app.route('/models/view/<int:id>/train_acc', methods=['GET'])
def view_train_acc(id):
    if request.method == 'GET' and model_eval['id'] == id:
        model = ModelLog.query.get_or_404(id)
        data = {'training_interval':model_eval['training_interval'],'training_acc': model_eval['training_accuracy']}
        return render_template('view_model_traina.html', model=model, data=data)

@app.route('/models/view/<int:id>/valid_acc', methods=['GET'])
def view_valid_acc(id):
    if request.method == 'GET' and model_eval['id'] == id:
        model = ModelLog.query.get_or_404(id)
        data = {'valid_interval':model_eval['validation_interval'],'valid_acc': model_eval['validation_accuracy']}
        return render_template('view_model_valida.html', model=model, data=data)

@app.route('/models/view/<int:id>/test_acc', methods=['GET'])
def view_test_acc(id):
    if request.method == 'GET' and model_eval['id'] == id:
        model = ModelLog.query.get_or_404(id)
        data = {'test_interval':model_eval['testing_interval'],'test_acc': model_eval['testing_accuracy']}
        return render_template('view_model_testa.html', model=model, data=data)

@app.route('/models/delete/<int:id>')
def delete(id):
    model = ModelLog.query.get_or_404(id)
    db.session.delete(model)
    db.session.commit()
    return redirect('/models')

@app.route('/models/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    model = ModelLog.query.get_or_404(id)
    if request.method == 'POST':
        model.model_name = request.form['model_name']
        model.model_type = request.form['model_type']
        model.creator = request.form['creator']
        model.description = request.form['description']

        model.train_ratio = request.form['train_ratio']
        model.test_ratio = request.form['test_ratio']
        model.train_batch_size = request.form['train_batch_size']
        model.test_batch_size = request.form['test_batch_size']
        model.epochs = request.form['epochs']
        model.lr = request.form['lr']
        model.mom = request.form['mom']
        model.reg = request.form['reg']

        model.loss_func = request.form['loss_func']
        model.opt = request.form['opt']
        model.rdm_seed = request.form['rdm_seed']
        model.log = request.form['log']
        model.opt_args = request.form['opt_args']

        db.session.commit()
        return redirect('/models')
    else:
        return render_template('edit.html', model=model)

@app.route('/models/new', methods=['GET', 'POST'])
def new_model():
    if request.method == 'POST':
        model_name = request.form['model_name']
        model_type = request.form['model_type']
        creator = request.form['creator']
        description = request.form['description']

        train_ratio = request.form['train_ratio']
        test_ratio = request.form['test_ratio']
        train_batch_size = request.form['train_batch_size']
        test_batch_size = request.form['test_batch_size']
        epochs = request.form['epochs']
        lr = request.form['lr']
        mom = request.form['mom']
        reg = request.form['reg']

        loss_func = request.form['loss_func']
        opt = request.form['opt']
        rdm_seed = request.form['rdm_seed']
        log = request.form['log']
        opt_args = request.form['opt_args']

        new_model = ModelLog(model_name=model_name, model_type=model_type, description=description, creator=creator, 
        train_ratio=train_ratio, test_ratio=test_ratio, train_batch_size=train_batch_size, test_batch_size=test_batch_size, epochs=epochs,
        lr=lr, mom=mom, reg=reg, loss_func=loss_func, opt=opt, rdm_seed=rdm_seed, log=log, opt_args=opt_args)
        db.session.add(new_model)
        db.session.commit()
        return redirect('/models')
    else:
        return render_template('new_model.html')

if __name__ == "__main__":
    app.run(debug=True)