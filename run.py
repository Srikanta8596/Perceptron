import pandas as pd
import os
from utils.all_utils import prepare_data,save_plot
from utils.models import Perceptron
import logging 
log_dir='log'
gate='OR gate'
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir,"running.log"),
    level=logging.INFO,
    format = '[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode='a'
)

def main(data,modelName,plotName,eta,epochs):
    df=pd.DataFrame(data)
    logging.info(f"This is the raw dataset: {df}")
    X,y = prepare_data(df)
    model= Perceptron(eta=eta,epochs=epochs)
    model.fit(X,y)

    _=model.total_loss()

    model.save(filename=modelName,model_dir='model')

    save_plot(df,model,filename=plotName)


if __name__=='__main__':
    OR ={
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,1]
    }
    
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    try:
        logging.info(f">>>>>>>>>>>>>>>Starting Training {gate}>>>>>>>>>>>>>>>>>>>")
        main(data=OR,modelName="or.model",plotName="or.png",eta=ETA,epochs =EPOCHS)
        logging.info(f"<<<<<<<<<<<<<<<<<<End Training {gate}<<<<<<<<<<<<<<<<<<<<\n\n")
    except Exception as e:
        logging.exception(e)
        raise e


    
