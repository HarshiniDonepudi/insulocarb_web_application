from flask import Flask,render_template,url_for,request, send_from_directory, redirect, flash,send_file
from werkzeug.utils import secure_filename
import detectron2
import numpy as np
import shutil
import os
import cv2
import random
from detectron2.utils.visualizer import ColorMode
from matplotlib import colors, pyplot as plt
import json
import pycocotools
from pycocotools.coco import COCO
import requests
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer 
from detectron2.engine import DefaultPredictor
import os
from skimage import io
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



#UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = 'some_secret'


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, "static")

 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
      


           
@app.route('/', methods=['GET', 'POST'])
def upload_file():
		
    if request.method == 'POST':
		
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], filename),os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg') )
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image uploaded. Click detect and wait.  ')
            return redirect(url_for('upload_file'))
    
    
    
    return render_template('index.html')
    
    



app.route('/<filename>')
def send_image(filename):
    return send_from_directory("images", filename )
	


@app.route('/predict', methods= ['POST'])
def predict():
    train_coco=COCO("/Users/harshinisaidonepudi/Downloads/seg.v5i.coco-segmentation/valid/_annotations.coco.json")
    category_ids = train_coco.loadCats(train_coco.getCatIds())

    category_names = [_["name"] for _ in category_ids]
    print(", ".join(category_names))
    category_ids
    dataset_name = 'testing'

    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)

    register_coco_instances(dataset_name,{},'/Users/harshinisaidonepudi/Downloads/seg.v5i.coco-segmentation/valid/_annotations.coco.json', "/Users/harshinisaidonepudi/Downloads/seg.v5i.coco-segmentation/valid")  
    metadata = MetadataCatalog.get("testing")      
       


  
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ("testing",)
    cfg.MODEL.WEIGHTS = '/Users/harshinisaidonepudi/Downloads/output-2 2/model_final.pth'# Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 34
  
    cfg.MODEL.DEVICE = "cpu"
    model= build_model(cfg)
  
    cfg.MODEL.WEIGHTS = '/Users/harshinisaidonepudi/Downloads/output-2 2/model_final.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST <= 0.7   # set the testing threshold for this model

    predictor = DefaultPredictor(cfg)

  
    im= cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg"))
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata, scale=0.8,instance_mode=ColorMode.IMAGE)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite((os.path.join(app.config['UPLOAD_FOLDER'], "prediction.jpg")),v.get_image()[:, :, ::-1])
    dict = outputs
    print(dict)
    a=outputs["instances"]
    b=a.get_fields()
    scores=b['scores'].cpu().numpy()
    rois=b['pred_boxes'].tensor.cpu().numpy()
    pred_class=b['pred_classes'].cpu().numpy()
    # pred_masks=b['pred_masks'].cpu().numpy()
    min_conf=0.7
    classtype = np.array(pred_class)
    classtype=classtype.astype(int)
    image_list=[]
    image_names=[]
    confidence=[]
  
    
    for i in range(len(classtype)):
        if scores[i]>= min_conf and category_names[pred_class[i]]!="Background":
            # temp= io.imread('/Users/harshinisaidonepudi/Downloads/Fashion-Detection-DeepFashion2-Detectron2-Flask-master/static/image.jpg')
            # for j in range(temp.shape[2]):
            #     temp[:,:,j] = temp[:,:,j] * masking[i,:,:]
            #     temp=cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            #     globals()[f"temp_{i}"] = temp
                image_list.append(i)
                # b=cv2.countNonZero(temp)
                
                confidence_score = f"Confidence: {100*scores[i]}  ,    Class: {category_names[pred_class[i]]}"
                confidence.append(confidence_score)
                image_names.append(category_names[pred_class[i]])
                # cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{i}.png"),temp)
    image_data=[]
    for i in range(len(image_list)):
        image_data.append({
         'name': image_names[i],
         'confidence': confidence[i],
         

     })
    print(image_data)
    flash('Successful! Click Show to view. Press Clear before next use of the detector')
    return render_template('index.html',prediction = v.get_image()[:, :, ::-1],data=image_data, names=image_names)


@app.route('/about')
def about():
	return render_template('about.html')
	
@app.route('/show')
def show():
   

    return render_template('show.html')
@app.route('/calculate',methods= ['POST'])
def calculate():
    image_names = predict()
    total_volume = []
    total_calorie = []
    total_carbs = []
    total_fats = []
    total_protien = []
    for i in range(len(image_names)):
        volume = request.form.get('names{}',format(i),'')
        total_volume.append(volume)
    print(total_volume)
    food_dic = {'aloo_methi': [100, 150, 12, 3, 10], 'aloo_tikki': [100, 157, 24.3, 3.6, 5],
            'bandar_laddu': [100, 390, 41.04, 5.63, 23.64],
            'besan_cheela': [100, 269, 36.61, 13.09, 8.07], 'biryani': [100, 139, 19.23, 6.36, 3.93],
            'butter_chicken': [100, 202, 4.32, 13.87, 14.52],
            'butter_naan': [100, 90, 44, 7, 1], 'chaat': [100, 189, 24.96, 4.8, 7.98],
            'chapati': [100, 170, 32.5, 5.84, 1.55],
            'chole': [100, 131, 13.3, 6, 6], 'dal': [100, 101, 13.36, 5.28, 3.23],
            'dal_makhani': [100, 181, 20.49, 6.78, 8.77],
            'dosa': [100, 212, 20.49, 6.78, 8.77], 'dum_aloo': [100, 131, 19.74, 3.19, 5.56],
            'gajar_ka_halwa': [100, 175, 30.48, 4.07, 5],
            'gulab_jamun': [100, 323, 39.26, 7.15, 15.75], 'idli': [100, 135, 26.31, 6.36, 0.62],
            'indian_bread': [100, 130, 19, 5, 3.5],
            'kulfi': [100, 227, 26.28, 4, 12.59], 'palak_paneer': [100, 169, 6.07, 7.89, 13.18],
            'paneer': [100, 293, 2.97, 14.01, 25.13],
            'pani_puri': [100, 307, 57.76, 6.09, 12.82], 'plain_rice': [100, 129, 89, 9, 2],
            'poha': [100, 110, 18.8, 2.34, 2.87],
            'poori': [100, 296, 46.73, 7.54, 9.43], 'rajma': [100, 165, 19.77, 7.04, 7.08],
            'rasgulla': [100, 213, 48.24, 3.62, 1.11],
            'sambar': [100, 513, 10.2, 2.8, 0.27], 'samosa': [100, 308, 32.21, 4.67, 17.86],
            'sheer_korma': [100, 208, 36.65, 4.2, 5.85],
            'upma': [100, 209, 38.06, 6.76, 3.15], 'uttapam': [100, 214, 30, 3.79, 8.76],
            'vada': [100, 282, 40.97, 10.59, 8.64]}
   

    for item in image_names:
        if item in food_dic:
            for i in range(len(total_volume)-1):
                calorie = (total_volume[i] / food_dic[item][0]) * food_dic[item][1]
                total_calorie.append(calorie)
                carbs = (total_volume[i] / food_dic[item][0]) * food_dic[item][2]
                total_carbs.append(carbs)
                fat = (total_volume[i] / food_dic[item][0]) * food_dic[item][3]
                total_fats.append(fat)
                protien = (total_volume[i] / food_dic[item][0]) * food_dic[item][4]
                total_protien.append(protien)
    print(total_calorie)
    print(total_carbs)
    print(total_fats)
    print(total_protien)
    nutrients = []
    meal_calories = sum(total_calorie)
    meal_carbs = sum(total_carbs)
    nutrients.append(meal_carbs)
    meal_fats = sum(total_fats)
    nutrients.append(meal_fats)
    meal_protiens = sum(total_protien)
    nutrients.append(meal_protiens)
    # nutrients = [meal_carbs, meal_fats, meal_protiens]
    # myexplode = [0.3, 0.2, 0.1]
    # plt.pie(nutrients,
    #         startangle=90,
    #         counterclock=False,
    #         autopct=lambda p: '{:.2f}%'.format(p),
    #         labels=["carbohydrate:"+str(round(meal_carbs,2)), "fats:"+str(round(meal_fats,2)), "protien:"+str(round(meal_protiens,2))],
    #         colors=["#ECB390", "#94B49F", "#FCF8E8"],
    #         explode=myexplode,
    #         shadow=True
    #         )
    # plt.axis('equal')
    # plt.title('Total Calories in your meal = '+str(round(meal_calories,2)),pad=0.6)
    # plt.suptitle("Nutrient Value of your meal:",)
    # plt.legend()
    # plt.savefig('/static/pie_chart.png')  # save the chart to a file
    return render_template('index.html',n=total_volume)

def show():
   

    return render_template('show.html')
	

	

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

