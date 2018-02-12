from SketchSvm import SketchSvm
import argparse

# https://docs.python.org/2/howto/argparse.html
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='pre-trained model file to use', reuired=True)

args = parser.parse_args()
model_file = args.model

svm = SketchSvm()

model = svm.load_model(model_file)

print("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))

svm.test_model(model, './test/*/*.png')