#!/usr/bin/python3

import unittest, random, utils
from simulator import fastCube
from simulator.fastCube import Cube as FastCube
from simulator.slowCube import Cube as SlowCube

class FastCubeTest(unittest.TestCase):
    def testInitialization(self):
        slow = SlowCube()
        modifiedStr = ''.join([random.choice(utils.defaultColors) for _ in range(6*3*3)])
        slowModified = SlowCube(modifiedStr)

        fast = FastCube()
        fastModified = FastCube(slowModified)

        assert self._linearize(slow) == self._linearize(fast)
        assert self._linearize(slowModified) == self._linearize(fastModified)
        assert fast.is_solved()
        assert not fastModified.is_solved()

    def testOps(self):
        slow = SlowCube()
        fast = FastCube()

        for op in fastCube.moveRemaps:
            slow.sequence(op)
            fast.sequence(op)
            assert self._linearize(slow) == self._linearize(fast)

    def testSequence(self):
        slow = SlowCube()
        fast = FastCube()
        fast.sequence('')
        assert self._linearize(slow) == self._linearize(fast)

        seq = 'R D Li B'
        slow.sequence(seq)
        fast.sequence(seq)
        assert self._linearize(slow) == self._linearize(fast)

    def _linearize(self, cube):
        return ''.join(str(cube).split())


if __name__ == '__main__':
    unittest.main()
