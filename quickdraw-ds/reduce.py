import numpy as np
import glob

for cat_filename in glob.glob('./*.npy'):
    images = np.load(cat_filename)

    test_images = images[0:9]
    np.save("../quickdraw-test/" + cat_filename, test_images)

    train_images = images[10:300]
    np.save("../quickdraw-train/" + cat_filename, train_images)

print("Quickdraw sketch export completed")