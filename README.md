# Bachelorareit zu AdversarialExamples

## Ergebnisse aus Versuchen mit verschiedenen Pixel und Epsilon Werten
Im Ordner 'bilder' liegt ein Unterverzeichnis liegt der Oderner 'Ergebnisse_Versuche'. In diesem liegen,
unterteilt nach den verschiedenen Pixel Werten. Zu jeden Pixel Wert gibt es für die Werte: 0.1, 0.01, 0.001, 
.0.2, 0.002, 0.03, 0.003, 0.04, 0.005 und 0.005 die Bilder der erzeugten Adversarial Examples, aus welchen
auch die genutzten Bilder in der Arbeit stammen.

## Anleitung zur Benutzung des Codes
Um das Projekt zu nutzen sollte es am besten im PyCharm als neues Projekt geöffnet werden. 
Das Hauptprogramm welches ausgeführt werden sollte ist torch3.py.
Bei der Ausführung wird ein Modell der logistischen Regression erstellt und trainiert und
anschließend 25 Adversarial Examples erzeugt und der Vergleich des Originals und des Adversarials geplottet
und als .jpg im Unterordner 'bilder' gespeichert. In der Bildunteschrift und dem Dateinamen muss bei 
Veränderung des Epsilon-Wertes dieser manuell geändert werden. Die Pixel größe wird automatisch übernommen.


### Ändern der Parameter Pixelgröße und Epsilon

#### Ändern der Pixelgröße
In der Datei logreg.py (/torchvision/models/logreg.py) habe ich die logistische Regression in PyTorch
realisiert. Um die Pixel-Größe für die Erzeugung der Adversarial Examples zu ändern muss hier in Zeile 12 der Wert 'size'
auf ide gewünschte Größe geändert werden. Außerdem muss dieser in der Datei torch3.py in Zeile 11 auch auf den
selben Wert gesetzt werden.

#### Ändern des Epsilon-Wertes
Die ursprüngliche Funktion der Foolbox Bibliothek hat immer den best möglichen Epsilon-Wert bestimmt und diesen 
genutzt. Da in der Arbeit aber ein fester Wert genutzt werden sollte wurde die Funktion dementsprechen angepasst.
In Zeilte 113 der Datei gradient.py(/foolbox/attacks/gradient.py) kann der gewünschte Epsilon-Wert angegeben werden.

