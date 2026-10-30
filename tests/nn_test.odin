package nn_tests

import "core:math/rand"
import "core:testing"
import nn ".."

@(test)
xor_converges :: proc(t: ^testing.T) {
	state := rand.create(u64(1))
	gen := rand.default_random_generator(&state)
	net: nn.Net(2, 3, 1)
	nn.net_init(&net, .Sigmoid, gen)

	inputs := [4][2]f64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	outputs := [4][1]f64{{0}, {1}, {1}, {0}}
	order := []int{0, 1, 2, 3}
	for _ in 0 ..< 20000 {
		rand.shuffle(order, gen)
		for i in order do nn.net_train_step(&net, inputs[i], outputs[i], [1]bool{true}, 0.5)
	}
	for i in 0 ..< 4 {
		out := nn.net_forward(&net, inputs[i])
		testing.expectf(t, abs(out[0] - outputs[i][0]) < 0.1, "xor %v -> %v", inputs[i], out[0])
	}
}

@(test)
linear_regression :: proc(t: ^testing.T) {
	state := rand.create(u64(2))
	gen := rand.default_random_generator(&state)
	net: nn.Net(1, 4, 1)
	nn.net_init(&net, .Linear, gen)
	// y = 2x - 0.5
	for _ in 0 ..< 4000 {
		x := rand.float64_range(-1, 1, gen)
		nn.net_train_step(&net, [1]f64{x}, [1]f64{2 * x - 0.5}, [1]bool{true}, 0.05)
	}
	for x in ([]f64{-0.8, 0, 0.6}) {
		out := nn.net_forward(&net, [1]f64{x})
		testing.expectf(t, abs(out[0] - (2 * x - 0.5)) < 0.1, "f(%v) = %v", x, out[0])
	}
}

@(test)
masked_output_gets_no_gradient :: proc(t: ^testing.T) {
	state := rand.create(u64(3))
	gen := rand.default_random_generator(&state)
	net: nn.Net(2, 3, 2)
	nn.net_init(&net, .Linear, gen)
	before_w := net.outputWeights
	before_b := net.outputBias
	for _ in 0 ..< 100 do nn.net_train_step(&net, [2]f64{1, -1}, [2]f64{5, 99}, [2]bool{true, false}, 0.1)
	for k in 0 ..< 3 do testing.expect_value(t, net.outputWeights[k][1], before_w[k][1])
	testing.expect_value(t, net.outputBias[1], before_b[1])
	testing.expect(t, net.outputBias[0] != before_b[0])
}

@(test)
l2_shrinks_without_signal :: proc(t: ^testing.T) {
	state := rand.create(u64(4))
	gen := rand.default_random_generator(&state)
	net: nn.Net(3, 4, 1)
	nn.net_init(&net, .Linear, gen)
	size :: proc(net: ^nn.Net(3, 4, 1)) -> (s: f64) {
		for row in net.hiddenWeights do for w in row do s += w * w
		return
	}
	before := size(&net)
	for _ in 0 ..< 500 do nn.net_train_step(&net, [3]f64{0, 0, 0}, [1]f64{0}, [1]bool{false}, 0.1, 0.1)
	testing.expect(t, size(&net) < before * 0.5)
}

@(test)
deterministic :: proc(t: ^testing.T) {
	run :: proc() -> nn.Net(2, 3, 1) {
		state := rand.create(u64(7))
		gen := rand.default_random_generator(&state)
		net: nn.Net(2, 3, 1)
		nn.net_init(&net, .Linear, gen)
		for _ in 0 ..< 50 do nn.net_train_step(&net, [2]f64{0.3, -0.2}, [1]f64{1}, [1]bool{true}, 0.1)
		return net
	}
	a, b := run(), run()
	testing.expect(t, a.hiddenWeights == b.hiddenWeights && a.outputWeights == b.outputWeights)
}
