from SketchSvm import SketchSvm

svm = SketchSvm()

# model = svm.train_model("./img/**")
# model = svm.load_model("18-02-08_17:42:45")

googletestpath = './test_google/*.npy'

c_range = [1, 10, 100, 1000]
gamma_range = [.001, .01, 0.1]
model = svm.train_model_google(googletestpath, c_range, gamma_range, kernel="linear", num=250)

print("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))

# svm.test_model(model, './test/*/*150x150.png')
svm.test_google(model, googletestpath)
