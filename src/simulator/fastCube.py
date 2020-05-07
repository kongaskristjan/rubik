
import copy, torch, utils
from simulator.slowCube import Cube as SlowCube

class Cube:
    def __init__(self, slowCube=None):
        if slowCube is None:
            self.indices = initialCubeLayout.clone()
        else:
            self.indices = utils.strToIndices(str(slowCube))

    def sequence(self, moveStr):
        self.indices = self.indices.view(6*3*3)
        for name in moveStr.split():
            self.indices = self.indices[moveRemaps[name]]
        self.indices = self.indices.view(6, 3, 3)

    def __str__(self):
        return utils.indicesToStr(self.indices)

    def is_solved(self):
        return (self.indices == initialCubeLayout).all().item()


def _createRemap(move):
    # Create initCube, finalCube
    initStr = ''.join([chr(ord('A') + i) for i in range(6*3*3)])
    initCube = SlowCube(initStr)
    initIndices = utils.strToIndices(str(initCube), colors=initStr).view(6 * 3 * 3)

    finalCube = copy.deepcopy(initCube)
    finalCube.sequence(move)
    finalIndices = utils.strToIndices(str(finalCube), colors=initStr).view(6 * 3 * 3)

    # Create index remapping
    remap = torch.zeros(6*3*3, dtype=torch.long)
    for i in range(6*3*3):
        remap[i] = torch.nonzero(initIndices == finalIndices[i].item()).item()

    return remap


moveRemaps = {m: _createRemap(m) for m in ['R', 'Ri', 'L', 'Li', 'U', 'Ui', 'D', 'Di', 'F', 'Fi', 'B', 'Bi']}
initialCubeLayout = utils.strToIndices(str(SlowCube()))
