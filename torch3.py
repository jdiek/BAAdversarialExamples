import foolbox
import torch
import torchvision.models as models
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.transform import resize


size = 28 #size of image nxn pixel

# instantiate the model

logisticRegression = models.logreg(pretrained=False).eval()
fmodel = foolbox.models.PyTorchModel(
    logisticRegression, bounds=(0, 9), num_classes=10)

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()])

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='.',
                            train=True,
                            transform=transform,
                            download=True)

test_dataset = dsets.MNIST(root='.',
                           train=False,
                           transform=transform
                           )



erfolg = 0
miserfolg = 0

for i in range(0,25):
    image, label = test_dataset[i]
    probPrediction = (foolbox.utils.softmax(fmodel.predictions(image.numpy()))[label])
    fig = plt.figure(figsize=(20, 4))
    fig.suptitle('\n\n\n\n\n\n\n\n\n\n\n\nAdversarial Example für ein Bild aus dem MNIST Datensatz mit Epsilon = 0.001\n und einer Groessee von %i x %i Pixeln\n' % (size, size), fontsize=22)
    plt.subplot(1,3,1)
    plt.imshow(np.reshape(image, (size, size)), cmap=plt.cm.gray)
    plt.title('\nOriginal Bild: %i\n Wahrscheinlichkeit des Models für \ndieses Label: %8.2f %%' % (np.argmax(fmodel.predictions(image.numpy())),probPrediction*100), fontsize=20)

    label = np.argmax(fmodel.predictions(image.numpy()))
    # apply attack on source image
    attack = foolbox.attacks.FGSM(fmodel)
    adversarial = attack(image.numpy(), label)

    try:
        advLabel = np.argmax(fmodel.predictions(adversarial))
        probPrediction = (foolbox.utils.softmax(fmodel.predictions(adversarial))[advLabel])
        erfolg = erfolg +1
        plt.subplot(1, 3, 2)
        plt.imshow(np.reshape(adversarial, (size, size)), cmap=plt.cm.gray)
        plt.title('\nAdversarial Example(FGSM): %i\n  Wahrscheinlichkeit des Models für \ndieses Label: %8.2f %%' % (advLabel, probPrediction*100), fontsize=20)

        plt.subplot(1, 3, 3)
        plt.imshow(np.reshape(adversarial, (size, size)) - np.reshape(image, (size, size)), cmap=plt.cm.gray, interpolation='none')
        plt.title('\nAdversarial Example - Original Input\n', fontsize=20)
        plt.savefig('bilder/AdvEx_%i_Pixel%i_Epsilon=0,001.jpg' % (i,size),  bbox_inches='tight', dpi=fig.dpi)
        plt.show()
    except:
        miserfolg = miserfolg +1
        plt.subplot(1, 3, 2)
        plt.title('Adversarial Example(FGSM): \n kein Adversarial Example', fontsize=20)

        plt.savefig('bilder/AdvEx_%i_Pixel%i_Epsilon=0,001.jpg' % (i,size), bbox_inches='tight')
        plt.show()

print('FGSM war erfolgreich bei %i und unerfolgreich bei %i Bildern\n' %(erfolg, miserfolg))