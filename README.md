# Solving Rubik's Cube with neural networks

## Objective

This project solves the 3x3x3 Rubik's Cube with neural networks. The goal was to find the simplest/stupidest approach that actually works. Thus no game trees nor reinforcement learning like DeepCube or DeepCubeA.

## Methology

Training is done by scrambling a solved cube anywhere from 1 to 20 times, and making the neural network predict to undo the last step using the cube state.

Thus, if a scrambled cube is given, it is likely to predict moves that will unscramble the cube.

## Details

Neural network details:
* Input: 6 * 6 * 9 (6 neurons per tile)
* Output: 12 (only 90 deg. rotations)
* Architecture: 6 * 6 * 9 -> 8192 -> 4096 -> 1024 -> 6 * 2 (similar to DeepCube)

## Performance:

__accuracy@step vs epochs (log plot) (even steps are omitted for clarity).__

![Training accuracy vs epochs](/images/training_accuracy.png)

The solver has so far solved 100% of cubes, however it takes on average 4600 moves to solve a fully scrambled cube.

## Is it just overfitting?

Rubik's cube has roughly 4x10^19 states, but the nn. sees roughly 6x10^7 states during it's training. If the actions of the nn. were fully random on unseen states, the probability of stumbling on a visited state during the 4600 steps is p<10^-8.

Thus no, it learns some useful generalizable knowledge.