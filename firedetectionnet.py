from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

class FireDetectionNet:
    @staticmethod
    def build(width , height , depth , classes):

        model = Sequential()
        inputShape = (height , width , depth)
        chanDim = -1
        
        ## CONV => RELU => POOL
        model.add(SeparableConv2D(16 ,(7,7),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2,2)))

        ##CONV => RELU => POOL
        model.add(SeparableConv2D(32 , (3,3) , padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2,2)))

        # (CONV => RELU) * 2 => POOL
        model.add(SeparableConv2D(64 , (3,3) , padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(SeparableConv2D(64 , (3,3) , padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2,2)))

        #first set of FC => Relu
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        #second set of FC => Relu
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model