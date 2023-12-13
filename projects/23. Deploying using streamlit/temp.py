import numpy as np
import pickle
import tensorflow as tf
import streamlit as st


# loading the saved model
loaded_model = tf.keras.models.load_model('C:/Users/karba/OneDrive/Desktop/code/projects/Deploying using streamlit/my_model.keras')
# loaded_model = pickle.load(open('C:/Users/karba/OneDrive/Desktop/code/projects/Deploying using streamlit/trained.pkl',"rb"))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.number_input('Number of Pregnancies')
    Glucose = st.number_input('Glucose Level')
    BloodPressure = st.number_input('Blood Pressure value')
    SkinThickness = st.number_input('Skin Thickness value')
    Insulin = st.number_input('Insulin Level')
    BMI = st.number_input('BMI value')
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    Age = st.number_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()