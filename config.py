import os 

Base_dir = os.path.dirname(os.path.abspath(__file__))

fire_Path = os.path.join(Base_dir , 'data\\FIRE-SMOKE-DATASET')
Classes = ['Fire' , 'Neutral' , 'Smoke']
INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 100
MODEL_PATH = os.path.join(Base_dir,'src' , 'fire_detection_model')
OUTPUT_IMAGE_PATH = os.path.sep.join(["output", "examples"])
SAMPLE_SIZE = 50