import landmark_generator as lg
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time


def classificationSleepOrNonsleep(Y_Data_30_Frames):
    ## tensor generate
    Y_Data_30_Frames = tf.expand_dims(Y_Data_30_Frames, axis = 0)
    Y_Data_30_Frames = tf.expand_dims(Y_Data_30_Frames, axis = 3)

    h = model.predict(Y_Data_30_Frames)
   

    print(f"Sleep [{(h[0][1]*100)//1}%]") if h.argmax() == 1 else print(f"NO Sleep [{(h[0][0])*100//1}%]")
    



if __name__=="__main__":
    model = "models/output.h5"
    model = load_model(model)
    model.summary() # model Info
    
    startTime = time.time()
    frameCnt = 0
    for Y_Data_30_Frames in lg.run(visualize=1, max_threads=4, capture=0, model=-3):
        frameCnt += 1
        classificationSleepOrNonsleep(Y_Data_30_Frames)

        # FPS Calcurator
        '''
        presendTime = time.time() - startTime 
        if presendTime > 6:
            print(f"Tracking time : {presendTime:.3}")
            print(f"Frames : {frameCnt*30}")
            print(f"FPS : {frameCnt*30/presendTime:.3}")
        '''
    
