from SketchNet import SketchNet

from cnnmodels.SketchANetModel import SketchANetModel
from cnnmodels.SketchANetModelAdapted import SketchANetModelAdapted
from cnnmodels.FashionModel import FashionModel

# modeltype = SketchANetModel
# modeltype = SketchANetModelAdapted
modeltype = FashionModel

net = SketchNet(modeltype, quickdraw=True).train()
