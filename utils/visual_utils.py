import matplotlib.pyplot as plt

def save_img(image, label):
    imagen = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(imagen)
    plt.title(label)
    plt.axis('off')  
    plt.savefig('image_test.png')
