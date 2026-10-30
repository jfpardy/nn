package nn

import "core:math"
import "core:math/rand"

// A very small fully connected network: inputs -> one sigmoid hidden layer
// -> outputs. Sizes are compile-time, everything is fixed arrays, nothing
// allocates. Training is plain per-sample gradient descent (backprop), with
// optional L2 weight decay and per-output masks for missing targets.

Output_Kind :: enum {
	Sigmoid, // outputs in 0..1, e.g. classification like XOR
	Linear, // outputs unbounded, for regression
}

Net :: struct($IN, $HID, $OUT: int) {
	output_kind: Output_Kind,

	hiddenWeights: [IN][HID]f64,
	hiddenBias:    [HID]f64,
	outputWeights: [HID][OUT]f64,
	outputBias:    [OUT]f64,

	// Activations from the last forward pass, used by backprop.
	hiddenLayer: [HID]f64,
	outputLayer: [OUT]f64,
}

sigmoid :: proc(x: f64) -> f64 {
	return 1 / (1 + math.exp(-x))
}

// Derivative of the sigmoid, given its output (not its input).
dsigmoid :: proc(y: f64) -> f64 {
	return y * (1 - y)
}

// Small weights centred on zero (+-1/sqrt(fan in)), zero biases.
net_init :: proc(net: ^Net($IN, $HID, $OUT), kind: Output_Kind, gen := context.random_generator) {
	net^ = {}
	net.output_kind = kind
	hidden_range := 1 / math.sqrt(f64(IN))
	for i in 0 ..< IN {
		for j in 0 ..< HID {
			net.hiddenWeights[i][j] = rand.float64_range(-hidden_range, hidden_range, gen)
		}
	}
	output_range := 1 / math.sqrt(f64(HID))
	for i in 0 ..< HID {
		for j in 0 ..< OUT {
			net.outputWeights[i][j] = rand.float64_range(-output_range, output_range, gen)
		}
	}
}

net_forward :: proc(net: ^Net($IN, $HID, $OUT), input: [IN]f64) -> [OUT]f64 {
	// Hidden layer activation
	for j in 0 ..< HID {
		activation := net.hiddenBias[j]
		for k in 0 ..< IN {
			activation += input[k] * net.hiddenWeights[k][j]
		}
		net.hiddenLayer[j] = sigmoid(activation)
	}
	// Output activation
	for j in 0 ..< OUT {
		activation := net.outputBias[j]
		for k in 0 ..< HID {
			activation += net.hiddenLayer[k] * net.outputWeights[k][j]
		}
		net.outputLayer[j] = net.output_kind == .Sigmoid ? sigmoid(activation) : activation
	}
	return net.outputLayer
}

// One step of backprop on a single sample. Outputs with mask[j] == false have
// no target and contribute no error. `l2` decays every weight a little each
// step (biases are left alone). Returns the squared error of the masked outputs.
net_train_step :: proc(
	net: ^Net($IN, $HID, $OUT),
	input: [IN]f64,
	target: [OUT]f64,
	mask: [OUT]bool,
	lr: f64,
	l2: f64 = 0,
) -> (sq_error: f64) {
	net_forward(net, input)

	// Change in output
	deltaOutput: [OUT]f64
	for j in 0 ..< OUT {
		if !mask[j] do continue
		error := target[j] - net.outputLayer[j]
		sq_error += error * error
		deltaOutput[j] = net.output_kind == .Sigmoid ? error * dsigmoid(net.outputLayer[j]) : error
	}

	// Change in hidden layer
	deltaHidden: [HID]f64
	for j in 0 ..< HID {
		error := 0.0
		for k in 0 ..< OUT {
			error += deltaOutput[k] * net.outputWeights[j][k]
		}
		deltaHidden[j] = error * dsigmoid(net.hiddenLayer[j])
	}

	decay := 1 - lr * l2

	// Apply the changes in output weights. A masked output still decays, but
	// only through l2; its weights get no gradient.
	for j in 0 ..< OUT {
		net.outputBias[j] += deltaOutput[j] * lr
		for k in 0 ..< HID {
			net.outputWeights[k][j] = net.outputWeights[k][j] * decay + net.hiddenLayer[k] * deltaOutput[j] * lr
		}
	}
	// Apply the changes in hidden weights
	for j in 0 ..< HID {
		net.hiddenBias[j] += deltaHidden[j] * lr
		for k in 0 ..< IN {
			net.hiddenWeights[k][j] = net.hiddenWeights[k][j] * decay + input[k] * deltaHidden[j] * lr
		}
	}
	return
}
