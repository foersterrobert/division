###############################################################
#
#  Slide-Ruler
#  per KI (hier Neuronales Netz) lernen.
#  Hyperparameter-Tuning
# 
# Capgemini
#  
# 
# 
#  04.06.2021 Version 0.1 Matthias Penzlin
#
#
#
#
###############################################################
import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD 
from keras.initializers import VarianceScaling 
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu, elu
import sys
import talos as ta
import smtplib, ssl
from email.message import EmailMessage

#######################################
# E-Mail vorbereiten, die abgesendet wird, sobald das Programm fertig ist
# Achtung: Das Versender-Postfach muss den Zugriff per SMTP erlauben.
# Dies ist z.B.: bei GMX standardmäßig nicht der Fall !!!!
'''pwd = input('Type your password and press enter: ')

msg = EmailMessage()
msg.set_content('Hyperparameter-Tuning abgeschlossen')
msg["Subject"] = 'Hyperparameter-Tuning abgeschlossen'
msg["From"] = 'matthias.penzlin@gmx.de'
msg["To"]   = 'matthias.penzlin@sdm.de'

context=ssl.create_default_context()
'''


########################################
# Ausgabedatei-Name erzeugen
#  
now = datetime.datetime.now() 
output_file_name = "C:/Arbeit/Python/Slide-Ruler/NN_OUT_"+now.strftime("%Y%m%d%H%M%S")+".txt"
#####################
# Eingabedatei festlegen
#####################
input_file_name = "C:/Arbeit/Python/Slide-Ruler/Beispiele.csv"

print(output_file_name)


np.set_printoptions(edgeitems =1000)
###############################################
#Zufallswerte auf einen Anfang setzen (funktioniert aber nur teilweise => Ergebnisse sind nicht 100%ig wiederholbar.
np.random.seed(42)
##################################
#Datei öffnen
fFloat = open(input_file_name,"r")
XundY = np.loadtxt(fFloat, delimiter=";",skiprows=1)
fFloat.close()
			
#Die Daten in Ein- und Ausgang unterteilen

X=XundY[:,1:195]
Y=XundY[:,195:196]


#####################
# Alternative
# mit und ohne Normieren versuchen
#####################
#Daten normieren
sc=StandardScaler()
X= sc.fit_transform(X)

#Datum, welches zwischen Trainings- und Test-Daten unterscheidet definieren
# zwischen Trainings- und Test-Daten unterscheiden 
TestSet     = np.random.choice(X.shape[0],int(X.shape[0]*(1-0.80)), replace=False)
TrainSet    = np.delete(np.arange(0, len(Y) ), TestSet) 


x = X[TrainSet,:] 
y = Y[TrainSet]

XTest = X[TestSet,:]
YTest = Y[TestSet]


#####################
# Alternative Parameter
# optimizers:  'SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam'
# use_bias: Ob das Neuronale Netzwerk auch mit einem Offset arbeiten soll
# activation: 'softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','exponential','linear'
# batch_size: wieviele Daten gleichzeitig verarbeitet und deren Ergebnis "gemittelt" wird
# epochs: Anzahl der Durchläufe aller Beispiele
# metrics: A metric is a function that is used to judge the performance of your model: 'mean_absolute_error','accuracy'
# Dropout: Zu welchem Anteil immer mal wieder einfach die Daten durch 0 ersetzt werden soll
#####################


p = {
    'first_neuron': [11],
    'second_neuron': [7],
    'batch_size': [14],
    'dropout': [0.0],
    #'optimizers': ['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam'],
    'optimizers': ['Nadam'],
    'first_activation':['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','exponential','linear'],
    'second_activation':['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','exponential','linear'],
     'epochs': [50]
    }


def befund(x_train, y_train, x_val, y_val, params):
    np.random.seed(42)
    # replace the hyperparameter inputs with references to params dictionary 
    model = Sequential()
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['first_neuron'],input_dim=x_train.shape[1],kernel_initializer=VarianceScaling(), activation=params['first_activation'],use_bias=True))
    model.add(Dense(params['second_neuron'],kernel_initializer=VarianceScaling(), activation=params['second_activation'],use_bias=True))
    model.add(Dense(1,kernel_initializer=VarianceScaling(), activation='sigmoid',use_bias=True))
    model.compile(loss='mean_squared_error', optimizer=params['optimizers'], metrics=['accuracy'])
    # Ausgabe, damit man weiß warum es so lange dauert...
    print(",",params['first_neuron'],",",params['second_neuron'],",",params['batch_size'],",",params['dropout'],",",params['optimizers'],",",params['first_activation'],",",params['second_activation'],",",params['epochs'])    
   
    # make sure history object is returned by model.fit()
    out = model.fit(x, y,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=[XTest, YTest],
                    verbose=0)
       
    yp = model.predict(XTest)
    yp = yp.reshape(yp.shape[0])
                    
    Y_soll =0
    Y_ist =0

    summe_fehler_gegen_predicted =0
    summe_fehler_gegen_durchschnitt =0

    summe_Befund =0
    durchschnittlicher_Befund_anteil =0

    YTest_index = 0                
    for i in  np.nditer(yp):
        Y_soll=YTest[YTest_index]
        summe_Befund = summe_Befund + Y_soll
        YTest_index +=1                

    durchschnittlicher_Befund_anteil = summe_Befund/YTest_index


    YTest_index = 0
    for i in  np.nditer(yp):
        Y_ist=i
        Y_soll=YTest[YTest_index]
        #print("YTest_index:",YTest_index   ,"      Soll: ",Y_soll,"     Predicted: ",Y_ist)
        summe_fehler_gegen_predicted = summe_fehler_gegen_predicted + abs(Y_ist-Y_soll)
        summe_fehler_gegen_durchschnitt = summe_fehler_gegen_durchschnitt + abs(durchschnittlicher_Befund_anteil-Y_soll)
        YTest_index +=1

        
    Abweichung_gegen_predicted = summe_fehler_gegen_predicted/YTest_index
    Abweichung_gegen_durchschnitt = summe_fehler_gegen_durchschnitt/YTest_index

    print('# ---------------------------------------------------------') 
    print("durchschnittliche Abweichung gegen durchschnitt",Abweichung_gegen_durchschnitt)
    print("durchschnittliche Abweichung gegen predicted",Abweichung_gegen_predicted)
    print('# ---------------------------------------------------------') 

    del yp
    f = open(output_file_name, "a")
    f.write(str(params['first_neuron'])   +","+    
    str(params['second_neuron'])  +","+     
     str(params['batch_size'])  +","+     
    str(params['dropout'])  +","+     
    str(params['optimizers'])  +","+     
    str(params['first_activation'])  +","+     
    str(params['second_activation'])  +","+     
    str(params['epochs'])  +","+     
    str(Abweichung_gegen_durchschnitt).replace('[','').replace(']','')   +","+     
    str(Abweichung_gegen_predicted).replace('[','').replace(']','')  +
    "\n")

    f.close()
    # modify the output model
    return out, model



f = open(output_file_name, "a")
f.write(input_file_name+"\n")
f.write("First_Neuron,Second_neuron,batch_size,dropout,optimizers,first_activation,second_activation,epochs,Abweichung_gegen_durchschnitt,Abweichung_gegen_predicted\n")
f.close()

t = ta.Scan(x, y, p, befund)  


##################################
#E-Mail versenden, da fertig

'''with smtplib.SMTP('mail.gmx.net', port=587) as smtp:
    smtp.starttls(context=context)
    smtp.login(msg["From"], pwd)
    smtp.send_message(msg)'''