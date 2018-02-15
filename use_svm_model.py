from SketchSvm import SketchSvm
import argparse

MODE_QUICKDRAW = 'QUICKDRAW'

# https://docs.python.org/2/howto/argparse.html
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='pre-trained model file to use', required=True)
parser.add_argument('-q', '--quickdraw', help='if the model is trained with google quickdraw', action='store_const', const=MODE_QUICKDRAW)

args = parser.parse_args()
model_file = args.model
quickdraw = args.quickdraw == MODE_QUICKDRAW

if quickdraw:
    trainpath = './quickdraw-train/*.npy'
    testpath = './quickdraw-test/*.npy'
else:
    trainpath = './tu-train/**/'
    testpath = './tu-test/*/*.png'

svm = SketchSvm()
# needed for the scaler to find the right params
svm.fit_scaler(quickdraw, trainpath)
model = svm.load_model(model_file)

print("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))

svm.test_model(model, testpath)