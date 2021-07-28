# aktivierungsfunktion2

Division

Einem kleinen neuronalen Netz durch eine eigene Aktivierungsfunktion die Division beibringen

![aktivierungsfunktion2%20530533ff63aa4ec3b41e296da3c103d3/image1.png](aktivierungsfunktion2%20530533ff63aa4ec3b41e296da3c103d3/image1.png)

28. Juli 2021

Inhaltsverzeichnis

Theorie

1.1. Python-Funktion

Keras

2.1. Sigmoid Modell

2.2. Erstellung von Trainingsdaten

2.3. Einbau eigener Aktivierungsfunktion

2.4. Hyper-Parameter Optimierung

2.5. Eigene Differenzierung

2.6. Einspielen eigener Gewichte

3. Präsentation durch Streamlit

Theorie

Das Ziel des Projektes ist es, einem kleinen neuronalen Netz mit nur 2 Neuronen in der Zwischenschicht die Division beizubringen. Die Inputs sollen 2 positive Zahlen sein und als Output soll das Ergebnis / 100 herauskommen. Das Bild zeigt eine Möglichkeit von

![aktivierungsfunktion2%20530533ff63aa4ec3b41e296da3c103d3/image2.png](aktivierungsfunktion2%20530533ff63aa4ec3b41e296da3c103d3/image2.png)

Gewichten und Biases mit denen die Division funktioniert.

Da im feed-forward Vorgang eines neuronalen Netzes allerdings nur multipliziert und addiert wird ist es mit einer standardisierten Aktivierungsfunktion nicht möglich mit so wenigen Neuronen ein funktionierendes Divisionsmodel zu bauen. Was wir also brauchen ist eine Eigene Aktivierungsfunktion, die auf der einen Seite Exponentialrechnung nutzt und auf der anderen den Logarithmus.

![aktivierungsfunktion2%20530533ff63aa4ec3b41e296da3c103d3/image3.png](aktivierungsfunktion2%20530533ff63aa4ec3b41e296da3c103d3/image3.png)

1. Python Funktion
2. Als erstes erstellen wir eine einfach Python Funktion, um die Aktivierungsfunktion abzubilden.

Testen wir die Funktion mit dem Beispiel aus dem Bild.

2. Keras

2.1. Sigmoid Modell

Da wir uns jetzt der Theorie sicher sind, können wir als nächstes das einfache Neuronale Netz in Keras nachbauen. Da unser Output im Dezimalbereich ist bietet sich für den Anfang Sigmoid als Aktivierungsfunktion am besten an.

2.2. Trainingsdaten

Als nächstes brauchen wir Trainingsdaten die wir selber in Python erstellen können

Nun können wir das Modell mit unseren eigens erstellten Daten trainieren lassen und evaluieren.

Für den Anfang wählen wir eine batchsize von 32 und 90 Epochen. Als Optimizer nehmen wir adam und als Kostenfunktion den MAE (mean absolute error), da dieser für uns am besten zu Vergleichen ist.

Dieses Modell lernt zumindest und erzielt bessere Ergebnisse als durch strategisches Raten. Trotzdem Ist es immer noch sehr Stumpf.

2.3. Aktivierungsfunktion

Lass uns also im nächsten Schritt unsere eigene Aktivierungsfunktion in Keras definieren.

Nun können wir die eigenen Aktivierungsfunktion auf das Modell übertragen und schauen ob sich die Trainingsergebnisse verbessern.

Auch wenn wir so ein besseres Loss bekommen, ist nicht von einem Erfolg zu sprechen. Die Vorhersagen des Modells sind immer noch weit weg von wirklicher Division.

2.4. Hyper-Parameter

In der Hoffnung bessere Ergebnisse zu erzielen können wir die Hyper-Parameter optimieren. Dies tun wir über Talos. Wir können verschiedene Werte für die Parameter ausprobieren und am Ende das Zusammenspiel wählen mit dem niedrigsten MAE. Zusätzlich erstellen wir in unserem Modell eine Dropout Schicht.

Als optimales Ergebnis bekommen wir für den Optimizer nadam, als batchsize 4 und ein hohes Dropout von 0.0005. Gerade durch die Wahl des neuen Optimizers lässt sich das Loss weiter drücken. Trotzdem sind wir von dem Taschenrechner ähnlichen Ergebnis der theoretischen Python-Funktion noch weit entfernt.

2.5. Eigene Differenzierung

Aus irgendeinem Grund scheint die Backpropagation unseres Modells also noch nicht gut zu funktionieren. Lasst uns daher probieren, die Differenzierung unserer Aktivierungsfunktion eigenhändig zu berechnen. Dies können wir tun über den tensorflow decorator @tf.custom_gradient

Aber wie erwartet ändert sich das Loss dadurch nicht.

2.6. Einspielen eigener Gewichte

Wenn es auf diesen Wegen nicht klappt können wir die Gewichte aus dem Vortrag auf die Schichten unseres Neuronalen Netzes übertragen.

Nun bekommen wir ein MAE, dass der Theoretischen Python Funktion in nichts mehr nachsteht.

3. Streamlit

Um unsere Arbeit zu präsentieren habe ich eine Web-App mit Streamlit erstellt, die es jedem erlaubt, seine eigenen Divisionen durch mehrere geladene Modelle laufen zu lassen.

![aktivierungsfunktion2%20530533ff63aa4ec3b41e296da3c103d3/image4.png](aktivierungsfunktion2%20530533ff63aa4ec3b41e296da3c103d3/image4.png)

Probiert es unter [https://share.streamlit.io/foersterrobert/division/DivisionStreamlit.py](https://share.streamlit.io/foersterrobert/division/DivisionStreamlit.py) gerne selbst aus.

Der Vollständige Code ist über [https://github.com/foersterrobert/division](https://github.com/foersterrobert/division) zu finden.
