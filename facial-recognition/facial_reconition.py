
import urllib,urllib2
import IPython.display as Disp
from functools import partial


import face_recognition
import cv2,os,pickle,random,sys,ntpath,re,scipy,time,itertools,math
from PIL import Image
#import skvideo.io
import tmdbsimple as tmdb
from pytvdbapi import api

import IPython.display as Disp
from multiprocessing import Pool
from IPython.display import *

import numpy as np
import pandas as pd
#%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Iterable,Counter
import pandas_ml as pdml
import seaborn as sns; sns.set()

from sklearn.utils import resample
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import label
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification
from sklearn.datasets import fetch_lfw_people
#from imblearn.over_sampling import *
#-------------------------------image download utility--------------------
data_path=None

def get_cast_name_tmdb(series_name='two and half man'):
    """
    return cast name from tmdb
    series_name : name of tv series
    """
    tmdb.API_KEY = 'd8eb79cd5498fd8d375ac1589bfc78ee'
    search = tmdb.Search()
    response = search.tv(query=series_name)
    tv1=tmdb.TV(id=response['results'][0]['id'])
    return [i['name'] for i in tv1.credits()['cast']]


def get_cast_name_tvdb(series_name='two and half man'):
    """
    return cast name from tvdb
    series_name : name of series
    """
    db = api.TVDB("05669A6CC3005169", actors=True, banners=True)
    result = db.search(series_name, "en")
    show = result[0]
    show.update()
    return show.Actors




#Downloading entire Web Document (Raw Page Content)
def download_page(url):
    """
        #return entire Web Document (Raw Page Content)
    """
    version = (3,0)
    cur_version = sys.version_info
    if cur_version >= version:     #If the Current Version of Python is 3.0 or above
        import urllib.request    #urllib library for Extracting web pages
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
            req = urllib.request.Request(url, headers = headers)
            resp = urllib.request.urlopen(req)
            respData = str(resp.read())
            return respData
        except Exception as e:
            print(str(e))
    else:                        #If the Current Version of Python is 2.x
        import urllib2
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
            req = urllib2.Request(url, headers = headers)
            response = urllib2.urlopen(req)
            page = response.read()
            return page
        except:
            return"Page Not found"


#Finding 'Next Image' from the given raw page
def _images_get_next_item(s):
    start_line = s.find('rg_di')
    if start_line == -1:    #If no links are found then give an error!
        end_quote = 0
        link = "no_links"
        return link, end_quote
    else:
        start_line = s.find('"class="rg_meta"')
        start_content = s.find('"ou"',start_line+1)
        end_content = s.find(',"ow"',start_content+1)
        content_raw = str(s[start_content+6:end_content-1])
        return content_raw, end_content


#Getting all links with the help of '_images_get_next_image'

def _images_get_all_items(page):
    items = []
    while True:
        item, end_content = _images_get_next_item(page)
        if item == "no_links":
            break
        else:
            items.append(item)      #Append all the links in the list named 'Links'
            time.sleep(0.1)        #Timer could be used to slow down the request for image downloads
            page = page[end_content:]
    return items


def downloaded_images(data_path,cast_list,series_name=""):
    """
    download images at data_path location 
    data_path : root folder path where subfolder will be created
    cast_list : list of searchable characters name for which images have to be downloaded
    series_name : optional name of series to refine search
    """


    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for cast in cast_list:
        ########### Edit From Here ###########

        #This list is used to search keywords. You can edit this list to search for google images of your choice. You can simply add and remove elements of the list.
        cast=re.sub(r'[^\x00-\x7F]+',' ', cast)
        search_keyword = [cast+" "+series_name]

        #This list is used to further add suffix to your search term. Each element of the list will help you download 100 images. First element is blank which denotes that no suffix is added to the search keyword of the above list. You can edit the list by adding/deleting elements from it.So if the first element of the search_keyword is 'Australia' and the second element of keywords is 'high resolution', then it will search for 'Australia High Resolution'
        keywords = ['']

        ########### End of Editing ###########

        ############## Main Program ############
        t0 = time.time()   #start the timer

        #Download Image Links
        i= 0
        while i<len(search_keyword):
            items = []
            iteration = "Item no.: " + str(i+1) + " -->" + " Item name = " + str(search_keyword[i])
            print (iteration)
            print ("Evaluating...for", cast)
            search_keywords = search_keyword[i]
            search = search_keywords.replace(' ','%20')

             #make a search keyword  directory
            try:
                os.makedirs(data_path+search_keywords)
            except Exception as e:
                if e.errno != 17:
                    raise   
                # time.sleep might help here
                pass

            j = 0
            while j<len(keywords):
                pure_keyword = keywords[j].replace(' ','%20')
                url = 'https://www.google.com/search?q=' + search + pure_keyword + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
                raw_html =  (download_page(url))
                time.sleep(0.1)
                items = items + (_images_get_all_items(raw_html))
                j = j + 1
            #print ("Image Links = "+str(items))
            print ("Total Image Links = "+str(len(items)))
            print ("\n")


            #This allows you to write all the links into a test file. This text file will be created in the same directory as your code. You can comment out the below 3 lines to stop writing the output to the text file.
            info = open(data_path+'output.txt', 'a')        #Open the text file called database.txt
            info.write(str(i) + ': ' + str(search_keyword[i-1]) + ": " + str(items) + "\n\n\n")         #Write the title of the page
            info.close()                            #Close the file

            t1 = time.time()    #stop the timer
            total_time = t1-t0   #Calculating the total time required to crawl, find and download all the links of 60,000 images
            print("Total time taken: "+str(total_time)+" Seconds")
            print ("Starting Download...")

            ## To save imges to the same directory
            # IN this saving process we are just skipping the URL if there is any error

            k=0
            errorCount=0
            while(k<len(items)):
                from urllib2 import Request,urlopen
                from urllib2 import URLError, HTTPError

                try:
                    req = Request(items[k], headers={"User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
                    response = urlopen(req,None,15)
                    if os.path.exists(data_path):
                        output_file = open(data_path+search_keywords+"/"+str(cast)+'_'+str(k+1)+".jpg",'wb')

                    data = response.read()
                    output_file.write(data)
                    response.close();

                    print("completed ====> "+str(k+1))

                    k=k+1;

                except IOError:   #If there is any IOError

                    errorCount+=1
                    print("IOError on image "+str(k+1))
                    k=k+1;

                except HTTPError as e:  #If there is any HTTPError

                    errorCount+=1
                    print("HTTPError"+str(k))
                    k=k+1;
                except URLError as e:

                    errorCount+=1
                    print("URLError "+str(k))
                    k=k+1;
                except Exception as e:
                    print(e)



            i = i+1

        print("\n")
        print("Everything downloaded!")
        print("\n"+str(errorCount)+" ----> total Errors")
        Disp.clear_output()

        #----End of the main program ----#


        # In[ ]:
#--------------------------------images display, and landmarks utility----------
def show_face(image_file):
    """
    image_file: numpy array
    output:detected Face
    """
    face_location=face_recognition.face_locations(image_file)
    print("I found {} face(s) in this photograph.".format(len(face_location)))
    for face in face_location:
        cv2.rectangle(image_file, (face[1],face[0]), (face[3],face[2]), (0,100,255), 2)
        plt.figure()
        plt.imshow(image_file)
        plt.show()

def show_landmarks(image_file):
    """
    image_file: numpy array
    output:landmark on the image  
    """
    face_landmarks_list=face_recognition.face_landmarks(image_file)
    for landmarks in face_landmarks_list:
        for landmark_name,landmark_location in landmarks.items():
            color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
            landmark_location=map(list,landmark_location)
            #print(landmark_location)
            #pil_image = Image.fromarray(image)
            landmark_location=np.array(landmark_location)
            #print(landmark_location)
            cv2.polylines(image_file,[landmark_location],0,color, 2)
        plt.figure()
        plt.imshow(image_file)
        plt.show()
def get_encoding(file_path=None,image=None):
    """
    file_path : full file path from which image has to be read from
    image : actual image 
    output: returns list of encoding, label of file
    """
    encoding=None
    label=None
    if file_path is not None:
        encoding=face_recognition.face_encodings(face_recognition.load_image_file(file_path))
    elif image is not None:
        encoding=face_recognition.face_encodings(image)
    else:
        print("no input passed")
    return [encoding,ntpath.basename(file_path.replace('.jpg',""))]

def get_stratified_sample(X,y,verbose=True,test_size=.2):
    """
    return stratified sampled X and y
    X : x matrix(input)
    y : y matrix(output)
    test_size : fration of total data in test set
    """
    sss = StratifiedShuffleSplit(n_splits=10, test_size=test_size, random_state=0)
    sss.get_n_splits(X, y)
    print(sss)       
    for train_index, test_index in sss.split(X, y):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return [X_train,X_test,y_train,y_test]

#get_name_only=lambda name:''.join([i for i in name if not i.isdigit()]).replace("_","").lower()
get_name_only=lambda name:re.sub('[^A-Za-z]+', '', name).lower()
get_name_only.__name__='get_name_only'

inv_map=lambda my_map: {v: k for k, v in my_map.items()}
inv_map.__name__='inverse_mapping'

class label_encoder():
    """
    label_dict : loading path of pickel, prepared dictionary can be used (pickled dictionary) 
    data_path : saving  path of  labels pickle file

    """
    def __init__(self,labels_dict=None,data_path='models/labels.pickle'):
        self.data_path=data_path
        if labels_dict is None:
            if os.path.exists(self.data_path) :  
                self.labels=pickle.load(open(self.data_path,'rb'))
            else:
                try:
                    os.mkdir(os.path.dirname(self.data_path))
                except:
                    pass
                self.labels={}
        else:
            self.labels=labels_dict
    def fit(self,x):
        """
        fit label encoder
        x:list or a single element which has to be encoded
        """
        if ((isinstance(x, Iterable)) & (type(x)!=str)):
            iter1=list(set(x)-set(self.labels.keys()))
            for i in iter1:
                self.labels[i]=len(self.labels.keys())+1
        else:
            if x not in self.labels.keys():
                self.labels[x]=len(self.labels.keys())+1 
    def transform(self,key):
        """
        transform a key to its label
        key: set(list/tuple) of elements for which values has to be retrieved 
        """
        l=[]
        if ((isinstance(key, Iterable))&(type(key)!=str)):
            print("its an iterable")
            for i in key:
                try:
                    l.append(self.labels[i])
                except Exception as e:
                    print("iterable error",e)
            return l
        else:
            try:
                return self.labels[key]
            except Exception as e:
                print("error",e)
    def save(self):
        """
        save a label encoder
        """
        try:
            pickle.dump(self.labels,open(self.data_path,'wb'),protocol=2)
        except:
            #os.mkdir(self.data_path)
            pickle.dump(self.labels,open(self.data_path,'wb'),protocol=2)
                

    #-----------------modeling------------------



# lbl_enc=label_encoder(data_path=data_path+'models/labels.pickle')
# labels=lbl_enc.labels

def get_label_charac_dict(directory_path=data_path):
    """
    It loads root directory and get their characters name and assign labels to them
    directory_path : path of parent directory 

    """
    print("inside get_label_charac_dict function")
    try:
        charac_names=pickle.load(open(directory_path+'charac_names.pickle','rb'))
    except:
        charac_names={}
    
    im_files=get_image_files(directory_path)
    for file_name in im_files.keys():
        charac_names[file_name]=get_name_only(file_name)
    lbl_enc=label_encoder(data_path=directory_path+'models/labels.pickle')
    lbl_enc.fit(charac_names.values())
    lbl_enc.save()
    labels=lbl_enc.labels
    pickle.dump(charac_names,open(directory_path+'charac_names.pickle','wb'),protocol=2)
    print("returning from get_label_charac_dict function")
    
    return {'charac_names':charac_names,'labels':labels}

def get_image_files(directory_path=data_path,return_only_paths=True):

    """
    return all jpg file path 
    directory_path : path of parent directory
    return_only_paths : if False return images else path only
    """
    print("inside get_image_files function")
    paths={}
    image_files={}
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for name in files:
            path=os.path.join(root, name)
            if name.endswith('.jpg'):
                #removing name,number_from_data
                file_name=name.replace(".jpg","")#root.split('/').pop()+name.replace(".jpg","")  
                #if file_name in file_name_filt:
                print(path,file_name)
                try:
                    if return_only_paths:
                        paths[file_name]=path
                    else:
                        image_files=face_recognition.load_image_file(path)
                except:
                    print("encoding error")

        Disp.clear_output()
    print("returning from get_image_files function")    
    if return_only_paths:
        return paths
    else:
        return image_files
def prepare_data(directory_path=data_path,l_threshold=20,r_threshold=None,dump_file_path=None):
    """
    returns a vector X(128 sized) and encoded label y [X,y]
    directory_path : source path of images folder, with each image have parent folder as label of it
    l_threshold : minimum images for a profile in images list (that are data required to have, otherwise ignored)
    r_threshold : maximum images for a profile in images list
    Autoresizing is done for any image more than 300 in max dimension
    minm_num: minimum number of images in a class
    dump_file_path: dumps a tuple of X,y
    """
    print("entering into prepare data")
    jsn=get_label_charac_dict(directory_path)
    charac_names=jsn['charac_names']
    labels=jsn['labels']
    encoding_files={}
    t=pd.DataFrame(list(charac_names.values()))[0].value_counts()
    t=t[t>=l_threshold]
    if r_threshold is not None:
        t=t[t<=r_threshold]
    t1=t.index
    file_name_filt=[]
    print('total unique matches with criteria',t1.shape)
    for k,v in charac_names.items():
        if v in t1:
            file_name_filt.append(k)
    del t1
    
#     for root, dirs, files in os.walk(directory_path, topdown=False):
#         for name in files:
#             path=os.path.join(root, name)
#             if name.endswith('.jpg'):
#                 #removing name,number_from_data
#                 file_name=root.split('/').pop()+name.replace(".jpg","")  
#                 if file_name in file_name_filt:
#                     print(path,file_name)
    im_files=get_image_files(directory_path)
    for file_name,path in im_files.items():
        try:
            #image=face_recognition.load_image_file(path)
            im=cv2.imread(path)
            image= np.flip(im,2).astype(np.uint8)
            image_res=image
            if max(image.shape[0:2]) > 300.0:
                image_res=scipy.misc.imresize(image,300.0/max(image.shape[0:2]))
            encoding_1=face_recognition.\
            face_encodings(image_res)
            if len(encoding_1)==1:
                encoding_files[file_name]=encoding_1

        except Exception as e:
            print("encoding error",e)
        Disp.clear_output()
                #charac_names[file_name]=charac_name
#     if dump_file_path is not None:
#         print('dumping output')
#         pickle.dump(encoding_files,\
#                     open(dump_file_path+'_encoded_file.pickle','wb'),protocol=2)
    l=list(encoding_files.keys())
    for k in l:
        if len(encoding_files[k])!=1:
            del encoding_files[k]
        else:
            encoding_files[k]=encoding_files[k][0]
    encoding_df=pd.DataFrame(encoding_files).T
    encoding_df['label_enc']=[labels[get_name_only(i)] for i in encoding_df.index]
    X=encoding_df.iloc[:,:128].values
    y=encoding_df['label_enc'].values
    if dump_file_path is not None:
        print('dumping output')
        pickle.dump([X,y],open(dump_file_path+'_encoded_file.pickle','wb'),protocol=2)
    print("returning prepare data")
    return [X,y]

def process_data(X,y,minm_num=30):
    """
    SMOTE and resampling
    X : input encoded vector
    y : output labels
    minm_num : minimum number per image  per label required in 
    """
    print("inside preprocessing function")
    df=pdml.ModelFrame(X,y)
    sampler=df.imbalance.over_sampling.SMOTE()
    sampled=df.fit_sample(sampler)
    total_classes=len(np.unique(y))
    if sampled.shape[0]/total_classes<minm_num:
        resampled_class=resample(sampled.iloc[:,1:].values,sampled.target.values,n_samples=2*minm_num*total_classes)
        sampled=pd.DataFrame(resampled_class[0])
        sampled['.target']=resampled_class[1]


    desampled=sampled.groupby('.target').apply(lambda x: pd.DataFrame(x).sample(n=minm_num))
    desampled.reset_index(drop=True,inplace=True)
    print("returning from preprocess data")
    return [desampled[list(range(128))],desampled['.target']]

def train_model(base_model,X,y,minm_image_process=None,threshold_accuracy=.9,classes=range(10),dump_file_path=None):
    """
    incremental training module
    returns a new model after partial fit on give data
    X=128 sized vector 
    y=labels of vectors
    minm_image_process='how many images of a specific label have to be trained, oversampling undersampling is done,  
    classes:number of that is going to be used in this model have to defined in advance
    """
    print("entering training module")
    [X_train,X_test,y_train,y_test]=get_stratified_sample(X,y,verbose=False)
    if minm_image_process is not None:
        [X_processed,y_processed]=process_data(X_train,y_train,minm_num=minm_image_process)
    else:
        [X_processed,y_processed]=[X_train,y_train]
    if dump_file_path is not None:
        pickle.dump([X_processed,y_processed],open(dump_file_path+'_resampled.pickle','wb'))
    accuracy=0
    idx=0
    while accuracy<threshold_accuracy:
        try:
            base_model.partial_fit(X_processed,y_processed)
        except Exception as e:
            print(e)
            base_model.partial_fit(X_processed,y_processed,classes=classes)
        y_pred=base_model.predict(X_test)
        accuracy=classification.accuracy_score(y_test,y_pred)
        print("accuracy in iteration ",idx+1,' is =',accuracy)
        idx+=1
        if idx>10:
            break
    print("returning from train module")    
    return base_model

        
    
def get_pred_on_frame(frame,model,data_path,return_labels=False):
    """
    provide prediction on a frame(image)
    model: classifier model
    data_path: loading relevent file from the source
    return_label : face name of the image 
    """
    labels=pickle.load(open(data_path+'models/labels.pickle','rb'))
    inv_labels=inv_map(labels)
    #frame=frame.mean(axis=2)
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations)>0:
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        print("number of faces detected",len(face_locations))
        face_names = []
        for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
            try:
                #print(face_encoding)
                match = model.predict(np.array(face_encoding).reshape([1,128]))[0]
                predict_probab=model.predict_proba(np.array(face_encoding).reshape([1,128]))[0]
                #bin_prob=math.exp(predict_probab[match])/sum([math.exp(i)for i in predict_probab])
                bin_prob=predict_probab[match-1]
                #bin_prob=(predict_probab[match]-np.mean(predict_probab))/np.std(predict_probab)
                print(match,inv_labels[match],bin_prob)
                face_names.append(inv_labels[match]+ '(p='+str(np.round(bin_prob,3))+')')
                #face_names.append(inv_labels[match]+ ' prediction probability='+str(1/(1+math.exp(-bin_prob))))
            except Exception as e:
                print(e)


        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    if return_labels:
        return frame,face_names
    else:
        return frame

def get_video_processed(data_path,model):
    """
    returns label appended frame (text on image)
    data_path : dumping path of data(video)
    model : classifier model
    """
    labels=pickle.load(open(data_path+'models/labels.pickle','rb'))
    inv_labels=inv_map(labels)
    input_movie = cv2.VideoCapture(video_path)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter(data_path+'_output.avi', fourcc, 29.97, (640, 360))

    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0

    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1
        print('reading frame ',frame_number)
        # Quit when the input video file ends
        if not ret:
            break

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations)>0:
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            print("number of faces detected",len(face_locations))
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                try:
                    #print(face_encoding)
                    match = model.predict(np.array(face_encoding).reshape([1,128]))[0]
                    print(match,inv_labels[match])
                    face_names.append(inv_labels[match])
                except Exception as e:
                    print(e)


            # Label the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if not name:
                    continue

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                
            # Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(frame)
        if frame_number%10==0:
            Disp.clear_output()
#                 display(Image(frame))
#                 plt.imshow(frame)
#                 plt.show()

    # All done!
    input_movie.release()
    output_movie.rfelease()
    #cv2.destroyAllWindows()

def get_frame(video_path):
    cap=cv2.VideoCapture(video_path)
    #num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(num_frames)
    ret=True
    while ret:
        ret,frame=cap.read()
        yield frame

# if __name__=="__main__":
#	data_path='/disk1/Notebooks/prateek/new_images/game of thrones/'
# 	lbl_enc=label_encoder(data_path=data_path+'models/labels.pickle')
# 	labels=lbl_enc.labels
# 	[X,y]=prepare_data(directory_path=data_path,l_threshold=10,dump_file_path=data_path+'got')
# 	clf_sgd=SGDClassifier(loss='log',n_jobs=7,shuffle=True,class_weight=None,warm_start=False,n_iter = np.ceil(10**6 / 600),average=True)
# 	clf_sgd=train_model(clf_sgd,X,y,process_flag=False,threshold_accuracy=0.85)
# 	pickle.dump(clf_sgd,open(path1+'models/sgd_classifier.pickle','wb'),protocol=2)
# 	vids='video_3.mp4'
# 	frms=get_frame(vids)
# 	cap=cv2.VideoCapture(vids)
# 	num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 	def get_pred_on_frame_1(frame):
#         return get_pred_on_frame(frame,clf_sgd)
#     p=Pool(7)
#     frames=p.map(get_pred_on_frame_1, itertools.islice(frms,0,5000,100))
#     p.close()