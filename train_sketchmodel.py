from SketchSvm import SketchSvm
import time

start_time = time.time()

svm = SketchSvm()

# model = svm.load_model("18-02-08_17:42:45")

c_range = [10, 100, 1000]
gamma_range = [0.00001, 0.0001, 0.001]
model = svm.train(False, "./tu-train/**/", c_range, gamma_range, kernel="linear")

print("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))

svm.test_model(model, './tu-test/*/*.png')

print("--- %0.2f minutes ---" % ((time.time() - start_time) / 60))