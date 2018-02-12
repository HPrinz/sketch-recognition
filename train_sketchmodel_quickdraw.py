from SketchSvm import SketchSvm
import time

start_time = time.time()

svm = SketchSvm()

quickdraw_path = './quickdraw-train/*.npy'
c_range = [1, 10, 100, 1000]
gamma_range = [.001, .01, .1]

model = svm.train_model_google(quickdraw_path, c_range, gamma_range, kernel="linear")
print("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))

quickdraw_path_test = './quickdraw-test/*.npy'
svm.test_google(model, quickdraw_path_test)

print("--- %0.2f minutes ---" % ((time.time() - start_time) / 60))