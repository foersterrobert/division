from altair.vegalite.v4.schema.core import Step
import streamlit as st
import numpy as np
from tensorflow import keras
import time
import math

st.set_page_config(
    page_title="Neural-Devision-Network",
    page_icon="➗",
)

st.markdown(
    '''
    <style>
        # MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        button {
            width: 100% !important;
        }
    </style>
    ''', unsafe_allow_html=True
)

def divisionHard(i1, i2):
    h1 = AktivierungsFunktionHard(i1 * 0.000001 - 0.000001)
    h2 = AktivierungsFunktionHard(i2 * 0.000001 - 0.000001)
    output = AktivierungsFunktionHard(h1 * 33.3333 + h2 * -33.3333 - 3.912023)
    return output

def AktivierungsFunktionHard(x):
    if x <= 0:
        return 1.359140915 * math.exp(x - 1)
    elif x > 15:
        return 1 - 1/(109.0858178 * x - 1403.359435)
    else:
        return 0.03 * math.log(1000000 * x + 1) + 0.5

framework = st.selectbox("Model:", options=['Keras', 'Python Function'])

with st.form('integers'):
    col1, col2 = st.columns(2)
    int1 = col1.number_input("Int1", value=20.7, step=1.0)
    int2 = col2.number_input("Int2", value=4.26, step=1.0)
    btn = st.form_submit_button('Divide')

    if btn:
        if framework == 'Keras':
            model = keras.models.load_model('./model/KerasHard.pth')
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            arr = np.array([[float(int1), float(int2)]])
            st.write(float(model.predict(arr))*100)

        else:
            x = divisionHard(float(int1), float(int2))*100
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            st.write(x)

        st.write('Actual Division:', float(int1)/float(int2))


st.markdown('---')

st.write('Keras Custom Grad')
st.code(
    '''
@tf.custom_gradient
def custom_activation(x):
    smallerEqualZero = tf.less_equal(x, tf.constant(0.0))
    greaterZero = tf.greater(x, tf.constant(0.0))
    greaterFiveteen = tf.greater(x, tf.constant(15.0))
    smallerEqualFiveteen = tf.less_equal(x, tf.constant(15.0))
    result = tf.where(smallerEqualZero, 1.359140915 * tf.math.exp(tf.where(smallerEqualZero, (x-1), 0)), 
            tf.where(greaterFiveteen, 1 - 1/(109.0858178 * x - 1403.359435), 
            0.03 * tf.math.log(tf.where(greaterZero, tf.where(smallerEqualFiveteen, (1000000 * x + 1), 0), 0)) + 0.5))

    def grad(dy):
        return dy * tf.where(smallerEqualZero, 1.359140915 * tf.math.exp(tf.where(smallerEqualZero, (x-1), 0)), 
            tf.where(greaterFiveteen, 501379254/(4596191*(501379254 * x / 4596191 - 280671887 / 200000)**2), 
            30000/(1000000*x+1)))

    return result, grad
    '''
)

st.write('Keras')
st.code(
    '''
def custom_activation(x):
    smallerEqualZero = tf.less_equal(x, tf.constant(0.0))
    greaterZero = tf.greater(x, tf.constant(0.0))
    greaterFiveteen = tf.greater(x, tf.constant(15.0))
    smallerEqualFiveteen = tf.less_equal(x, tf.constant(15.0))
    return tf.where(smallerEqualZero, 1.359140915 * tf.math.exp(tf.where(smallerEqualZero, (x-1), 0)), 
            tf.where(greaterFiveteen, 1 - 1/(109.0858178 * x - 1403.359435), 
            0.03 * tf.math.log(tf.where(greaterZero, tf.where(smallerEqualFiveteen, (1000000 * x + 1), 0), 0)) + 0.5))
    '''
)


st.write('PyTorch')
st.code(
    '''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.where(x <= 0, 1.359140915 * (x-1).exp(), torch.where(x > 15, 1 - 1/(109.0858178 * x - 1403.359435), 0.03 * (1000000 * x + 1).log() + 0.5))
        x = self.l2(x)
        x = torch.where(x <= 0, 1.359140915 * (x-1).exp(), torch.where(x > 15, 1 - 1/(109.0858178 * x - 1403.359435), 0.03 * (1000000 * x + 1).log() + 0.5))
        x = x*5
        return x
    '''
)
st.write('Python Function')
st.code(
    '''
def divisionHard(i1, i2):
    h1 = AktivierungsFunktionHard(i1 * 0.000001 - 0.000001)
    h2 = AktivierungsFunktionHard(i2 * 0.000001 - 0.000001)
    output = AktivierungsFunktionHard(h1 * 33.3333 + h2 * -33.3333 - 3.912023)
    return output

def AktivierungsFunktionHard(x):
    if x <= 0:
        return 1.359140915 * math.exp(x - 1)
    elif x > 15:
        return 1 - 1/(109.0858178 * x - 1403.359435)
    else:
        return 0.03 * math.log(1000000 * x + 1) + 0.5
    '''
)

st.markdown("---")
st.markdown(
        """
        <h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://robertfoerster.com/">Robert</a></h6>
        <br>
        <a href="https://github.com/foersterrobert/division" target='_blank'><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Logo.png" alt="Streamlit logo" height="20"></a>
        <a href="https://www.linkedin.com/in/rfoerster/" target='_blank' style='margin-left: 10px;'><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/LinkedIn_Logo.svg/1000px-LinkedIn_Logo.svg.png" alt="Streamlit logo" height="26"></a>
        """,
        unsafe_allow_html=True,
    )
