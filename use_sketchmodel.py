from SketchSvm import SketchSvm
import argparse

# https://docs.python.org/2/howto/argparse.html
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='pre-trained model file to use', required=True)

args = parser.parse_args()
model_file = args.model

svm = SketchSvm()

# needed for the scaler to find the right params
svm.load_images("./tu-train/**")
svm.get_training_data(False)

model = svm.load_model(model_file)

print("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))

svm.test_model(model, './test/*/*.png')