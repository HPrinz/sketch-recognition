from SketchSvm import SketchSvm
import time

start_time = time.time()

svm = SketchSvm(28, 7)

quickdraw_path = './quickdraw-train/*.npy'
c_range = [1, 10, 100]
gamma_range = [.0001, .001]

model = svm.train(True, quickdraw_path, c_range, gamma_range, kernel="rbf")
print("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))

quickdraw_path_test = './quickdraw-test/*.npy'
svm.test_google(model, quickdraw_path_test)

print("--- %0.2f minutes ---" % ((time.time() - start_time) / 60))
