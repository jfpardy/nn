package xor

import "core:fmt"
import "core:math/rand"
import nn "../.."

// The original demo: learn XOR with a 2-2-1 network.

numInput :: 2
numHiddenNodes :: 2
numOutputs :: 1
numTrainingSets :: 4

main :: proc() {
	state := rand.create(u64(1))
	gen := rand.default_random_generator(&state)

	net: nn.Net(numInput, numHiddenNodes, numOutputs)
	nn.net_init(&net, .Sigmoid, gen)

	training_inputs := [numTrainingSets][numInput]f64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	training_outputs := [numTrainingSets][numOutputs]f64{{0}, {1}, {1}, {0}}
	trainingSetOrder := []int{0, 1, 2, 3}

	lr := 0.5
	numberOfEpochs := 20000

	for epoch in 0 ..< numberOfEpochs {
		rand.shuffle(trainingSetOrder, gen)
		total := 0.0
		for i in trainingSetOrder {
			total += nn.net_train_step(&net, training_inputs[i], training_outputs[i], [1]bool{true}, lr)
		}
		if epoch % 2000 == 0 do fmt.printf("epoch %5d  error %.6f\n", epoch, total)
	}

	for i in 0 ..< numTrainingSets {
		out := nn.net_forward(&net, training_inputs[i])
		fmt.printf("%g xor %g = %.3f (want %g)\n", training_inputs[i][0], training_inputs[i][1], out[0], training_outputs[i][0])
	}
}
